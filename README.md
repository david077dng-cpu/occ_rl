# Occupancy Grid RL

A complete Reinforcement Learning (RL) training pipeline for holonomic wheel robots navigating occupancy grid worlds with obstacles and targets.

## Overview

This project implements a state-of-the-art RL training system featuring:

- **2D Occupancy Grid World**: Simulated environment with static and dynamic obstacles
- **Holonomic Robot Model**: 3-omniwheel (Kiwi drive) kinematics (LeKiwi-style)
- **Gymnasium Environment**: Standard RL interface with Dict observation space
- **CNN+MLP Policy Architecture**: Custom RLlib model combining grid CNN with scalar MLP
- **PPO Training**: Ray RLlib implementation with curriculum learning support
- **Comprehensive Evaluation**: Success rate, collision rate, trajectory analysis

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Ray 2.0+

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# For Mac M1/M2 users, install PyTorch separately:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### 1. Test the Environment

```bash
cd occupancy_grid_rl/envs
python test_basic.py
```

This runs unit tests for:
- Grid world simulation
- Robot kinematics
- Environment reset/step

### 2. Train a Policy

```bash
python -m occupancy_grid_rl.training.train_ppo \
    --exp-name=my_first_run \
    --num-iterations=500 \
    --num-envs=4 \
    --lr=3e-4
```

Training logs and checkpoints are saved to `~/ray_results/grid_nav/`.

### 3. Evaluate the Policy

```bash
python -m occupancy_grid_rl.evaluation.eval_policy \
    ~/ray_results/grid_nav/my_first_run/checkpoint_000500 \
    --num-episodes=100 \
    --save-metrics=eval_results.json
```

### 4. Visualize Results

```python
from occupancy_grid_rl.evaluation.visualize import create_trajectory_plot
import numpy as np

# Load trajectory
trajectory = np.load("trajectory.npz")["episode_0"]

# Create plot
fig = create_trajectory_plot(
    trajectory=trajectory,
    obstacles=[{"position": [3, 3], "radius": 0.5}],
    goal=np.array([8, 8]),
    save_path="trajectory.png"
)
```

## Architecture

### Directory Structure

```
occupancy_grid_rl/
├── envs/                       # Simulation environments
│   ├── __init__.py
│   ├── grid_world.py          # 2D occupancy grid world
│   ├── robot_kinematics.py    # Holonomic robot kinematics
│   ├── occupancy_grid_env.py  # Gymnasium environment
│   ├── test_basic.py          # Unit tests
│   └── test_env.py            # Full integration tests
├── policies/                   # RL policy models
│   ├── __init__.py
│   └── grid_nav_policy.py     # Custom RLlib CNN+MLP model
├── training/                   # Training scripts
│   ├── __init__.py
│   └── train_ppo.py           # PPO training with RLlib
├── evaluation/                # Evaluation tools
│   ├── __init__.py
│   ├── eval_policy.py         # Policy evaluation
│   └── visualize.py           # Trajectory visualization
├── utils/                     # Common utilities
│   ├── __init__.py
│   └── common.py              # Helper functions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Observation Space

The environment uses a `Dict` observation space:

```python
{
    "occupancy_grid": Box(0, 1, (32, 32), float32),     # Local 32x32 view
    "robot_pose": Box(-inf, inf, (3,), float32),       # [x, y, theta]
    "target_relative": Box(-1, 1, (2,), float32),       # [dx, dy] normalized
    "velocity": Box(-inf, inf, (3,), float32),          # [vx, vy, omega]
}
```

### Action Space

Continuous action space for holonomic control:

```python
Box(
    low=[-0.5, -0.5, -90],    # [vx, vy, omega]
    high=[0.5, 0.5, 90],
    dtype=float32
)
```

Units:
- `vx`: Forward velocity in m/s
- `vy`: Lateral velocity in m/s
- `omega`: Angular velocity in deg/s

### Reward Function

Mixed dense + sparse reward:

```python
reward = (
    -0.01 * step_penalty +                    # Small step penalty
    10.0 * (prev_dist - curr_dist) +           # Dense: distance improvement
    -0.1 * collision_penalty +                  # Collision penalty
    100.0 * goal_reached_bonus +                # Sparse: goal reached
    -0.001 * action_magnitude_penalty           # Regularization
)
```

## Training Tips

### Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `lr` | 3e-4 | Learning rate | Increase if slow convergence, decrease if unstable |
| `gamma` | 0.99 | Discount factor | Lower for short-horizon tasks, higher for long-horizon |
| `lambda` | 0.95 | GAE lambda | Lower for low bias, higher for low variance |
| `clip_param` | 0.2 | PPO clip | Keep at 0.2-0.3 for stability |
| `entropy_coeff` | 0.01 | Entropy bonus | Increase for exploration, decrease after initial training |
| `num_envs` | 4 | Parallel envs | Increase for faster training (limited by CPU) |

### Curriculum Learning

Enable curriculum learning to gradually increase difficulty:

```bash
python -m occupancy_grid_rl.training.train_ppo \
    --exp-name=curriculum_run \
    --curriculum \
    --num-iterations=1000
```

Curriculum progression:
- Level 0: 3 static, 1 dynamic obstacle
- Level 1: 4 static, 1 dynamic obstacle
- Level 2: 5 static, 2 dynamic obstacles
- ... (increases as policy improves)

### Monitoring Training

View TensorBoard logs:

```bash
tensorboard --logdir=~/ray_results/grid_nav
```

Key metrics to monitor:
- `episode_reward_mean`: Should increase over time
- `episode_len_mean`: Should decrease as agent becomes more efficient
- `entropy`: Should decrease as policy becomes more deterministic
- `policy_loss` and `value_loss`: Should stabilize

## Troubleshooting

### Common Issues

**ImportError: No module named 'gymnasium'**
```bash
pip install gymnasium
```

**Ray out of memory errors**
- Reduce `--num-envs`
- Reduce `--rollout-fragment-length`
- Reduce `--train-batch-size`

**Policy not improving**
- Check reward function in environment
- Increase `--entropy-coeff` for more exploration
- Check observation normalization
- Verify action bounds are appropriate

**Slow training**
- Increase `--num-envs` (if CPU available)
- Use GPU with `--num-gpus=1`
- Decrease `--train-batch-size` for faster iterations
- Simplify network architecture

## Citation

If you use this code in your research, please cite:

```bibtex
@software{occupancy_grid_rl,
  title={Occupancy Grid RL: RL Training Pipeline for Holonomic Robot Navigation},
  author={XRollout},
  year={2024},
  url={https://github.com/xrollout/occupancy-grid-rl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) for distributed RL training
- Uses [Gymnasium](https://gymnasium.farama.org/) for the environment interface
- Inspired by LeKiwi holonomic robot design from [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
