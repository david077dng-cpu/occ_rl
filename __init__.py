"""
Occupancy Grid RL

A complete RL training pipeline for holonomic wheel robots navigating
occupancy grid worlds with obstacles and targets.

Modules:
    envs: Gymnasium environments for grid navigation
    policies: RL policy models (PPO with custom CNN+MLP architecture)
    training: Training scripts and utilities
    evaluation: Policy evaluation and visualization
    utils: Common utilities

Example usage:

    >>> from occupancy_grid_rl.envs import OccupancyGridEnv
    >>> env = OccupancyGridEnv()
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    """Lazy load classes to avoid import errors when dependencies are missing."""
    if name == "OccupancyGridEnv":
        from occupancy_grid_rl.envs import OccupancyGridEnv
        return OccupancyGridEnv
    elif name == "HolonomicKinematics":
        from occupancy_grid_rl.envs import HolonomicKinematics
        return HolonomicKinematics
    elif name == "RobotConfig":
        from occupancy_grid_rl.envs import RobotConfig
        return RobotConfig

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "OccupancyGridEnv",
    "HolonomicKinematics",
    "RobotConfig",
]
