"""
Custom RLlib Policy Model for Grid Navigation

This module implements a custom TorchModelV2 for RLlib that combines:
- CNN encoder for occupancy grid (32x32 -> 256-dim)
- MLP encoder for scalar features (pose, target, velocity -> 64-dim)
- Combined features -> policy/value heads

Architecture:
    Grid (32x32)
       ↓
    Conv2d(1, 32, 3, stride=2, padding=1)  # 16x16
    ReLU
    Conv2d(32, 64, 3, stride=2, padding=1)  # 8x8
    ReLU
    Flatten
    Linear(64*8*8, 256)
    ReLU
       ↓
    Grid features (256-dim)

    Scalar input (8-dim: pose(3) + target(2) + velocity(3))
       ↓
    Linear(8, 64)
    ReLU
       ↓
    Scalar features (64-dim)

    Concat [grid_features, scalar_features]
       ↓
    Combined features (320-dim)
       ↓
    Linear(320, 256)
    ReLU
       ↓
    Shared features (256-dim)
       ↓                    ↓
    Policy head           Value head
    Linear(256, 128)      Linear(256, 128)
    ReLU                  ReLU
    Linear(128, action_dim*2)  Linear(128, 1)
    (mean, log_std)       (value)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class GridEncoder(nn.Module):
    """CNN encoder for occupancy grid."""

    def __init__(self, input_shape: Tuple[int, int] = (32, 32), output_dim: int = 256):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim

        # Calculate conv output size
        # 32x32 -> 16x16 -> 8x8
        conv_output_size = 64 * 8 * 8

        self.conv = nn.Sequential(
            # Input: (batch, 1, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output: (batch, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Output: (batch, 64, 8, 8)

            nn.Flatten(),
            # Output: (batch, 64*8*8 = 4096)
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, output_dim),
            nn.ReLU(),
        )

    def forward(self, grid: TensorType) -> TensorType:
        """
        Forward pass.

        Args:
            grid: (batch, 32, 32) or (batch, 1, 32, 32)

        Returns:
            features: (batch, output_dim)
        """
        # Add channel dimension if needed
        if grid.dim() == 3:
            grid = grid.unsqueeze(1)

        conv_out = self.conv(grid)
        features = self.fc(conv_out)
        return features


class ScalarEncoder(nn.Module):
    """MLP encoder for scalar features (pose, target, velocity)."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, scalar_input: TensorType) -> TensorType:
        """
        Forward pass.

        Args:
            scalar_input: (batch, input_dim) containing [pose(3), target(2), velocity(3)]

        Returns:
            features: (batch, hidden_dim)
        """
        return self.mlp(scalar_input)


class GridNavTorchModel(TorchModelV2, nn.Module):
    """
    Custom RLlib TorchModelV2 for grid navigation.

    Combines CNN grid encoder with MLP scalar encoder for policy/value heads.

    Args:
        obs_space: Observation space (Dict with grid, pose, target, velocity)
        action_space: Action space (Continuous)
        num_outputs: Number of policy outputs (2 * action_dim for mean/log_std)
        model_config: RLlib model config
        name: Model name
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get action dimension (num_outputs / 2 for mean/log_std)
        self.action_dim = num_outputs // 2

        # Grid encoder: 32x32 -> 256
        self.grid_encoder = GridEncoder(input_shape=(32, 32), output_dim=256)

        # Scalar encoder: 8-dim -> 64
        self.scalar_encoder = ScalarEncoder(input_dim=8, hidden_dim=64)

        # Combined feature dimension: 256 + 64 = 320
        combined_dim = 256 + 64

        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
        )

        # Policy head: outputs mean and log_std for continuous actions
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim * 2),  # mean and log_std
        )

        # Value head: outputs state value
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens
    ):
        """
        Forward pass for inference (policy evaluation).

        Args:
            input_dict: Input dictionary containing observations
            state: Hidden state (not used)
            seq_lens: Sequence lengths (not used)

        Returns:
            action_outputs: Policy outputs (mean and log_std)
            state: Updated state (empty)
        """
        # Extract observations from input dict
        obs = input_dict["obs"]

        # Process through encoders
        grid_features = self._process_observations(obs)

        # Get policy outputs
        policy_out = self.policy_head(grid_features)

        return policy_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Compute the value estimate for the current observation.

        This is called after forward() during training.
        """
        # The shared features are stored during forward pass
        if hasattr(self, '_shared_features'):
            return self.value_head(self._shared_features).squeeze(-1)
        else:
            # Fallback (shouldn't happen in normal training)
            return torch.zeros(1)

    def _process_observations(self, obs: Dict[str, TensorType]) -> TensorType:
        """
        Process observations through encoders.

        Args:
            obs: Dictionary containing:
                - occupancy_grid: (batch, 32, 32)
                - robot_pose: (batch, 3)
                - target_relative: (batch, 2)
                - velocity: (batch, 3)

        Returns:
            shared_features: (batch, 256) shared feature representation
        """
        # Extract observations
        grid = obs["occupancy_grid"]  # (batch, 32, 32)
        robot_pose = obs["robot_pose"]  # (batch, 3)
        target_relative = obs["target_relative"]  # (batch, 2)
        velocity = obs["velocity"]  # (batch, 3)

        # Encode grid
        grid_features = self.grid_encoder(grid)  # (batch, 256)

        # Concatenate scalar features: pose(3) + target(2) + velocity(3) = 8
        scalar_input = torch.cat([robot_pose, target_relative, velocity], dim=-1)
        scalar_features = self.scalar_encoder(scalar_input)  # (batch, 64)

        # Combine features
        combined = torch.cat([grid_features, scalar_features], dim=-1)  # (batch, 320)

        # Shared representation
        shared_features = self.shared_fc(combined)  # (batch, 256)

        # Store for value function
        self._shared_features = shared_features

        return shared_features


# Register the model with RLlib (will be done in training script)
# from ray.rllib.models import ModelCatalog
# ModelCatalog.register_custom_model("grid_nav_model", GridNavTorchModel)
