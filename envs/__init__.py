"""
Occupancy Grid RL Environments

This module provides environments for RL training of holonomic robots
in occupancy grid worlds.
"""

from .grid_world import GridWorld, CellType, Obstacle, StaticObstacle, DynamicObstacle
from .robot_kinematics import HolonomicKinematics, RobotConfig

# Lazy import for environments that require gymnasium
def __getattr__(name):
    if name == "OccupancyGridEnv":
        from .occupancy_grid_env import OccupancyGridEnv
        return OccupancyGridEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GridWorld",
    "CellType",
    "Obstacle",
    "StaticObstacle",
    "DynamicObstacle",
    "HolonomicKinematics",
    "RobotConfig",
    "OccupancyGridEnv",
]
