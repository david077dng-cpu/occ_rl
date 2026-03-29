"""
Visualization Tools for Grid Navigation

This module provides visualization tools for:
- Episode replay from saved trajectories
- Real-time rendering of episodes
- Trajectory plotting with heatmaps
- Comparison plots for multiple policies

Usage:
    # Visualize a saved trajectory
    python visualize.py --trajectory=./trajectory.npz

    # Render an episode with trained policy
    python visualize.py --checkpoint=/path/to/checkpoint --render

    # Generate comparison plots
    python visualize.py --metrics=./metrics1.json,./metrics2.json --compare
"""

import argparse
import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import matplotlib (optional dependency)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting functions disabled.")

# Try to import imageio for video creation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def create_trajectory_plot(
    trajectory: np.ndarray,
    obstacles: Optional[List[Dict]] = None,
    goal: Optional[np.ndarray] = None,
    world_bounds: Tuple[float, float, float, float] = (0, 10, 0, 10),
    title: str = "Robot Trajectory",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Create a 2D plot of robot trajectory.

    Args:
        trajectory: Array of shape (T, 3) with [x, y, theta] at each timestep
        obstacles: List of obstacle dictionaries with 'position', 'radius', 'type'
        goal: Goal position [x, y]
        world_bounds: (xmin, xmax, ymin, ymax)
        title: Plot title
        save_path: Path to save figure (None = don't save)
        show: Whether to display the plot

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return None

    fig, ax = plt.subplots(figsize=(10, 10))

    xmin, xmax, ymin, ymax = world_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    if obstacles:
        for obs in obstacles:
            pos = obs['position']
            radius = obs.get('radius', 0.5)
            obs_type = obs.get('type', 'static')

            color = 'gray' if obs_type == 'static' else 'orange'
            circle = Circle(pos, radius, color=color, alpha=0.5)
            ax.add_patch(circle)

    # Plot goal
    if goal is not None:
        goal_circle = Circle(goal, 0.5, color='green', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')

    # Plot trajectory with color gradient
    if len(trajectory) > 1:
        positions = trajectory[:, :2]  # Extract x, y

        # Create color gradient from start (blue) to end (red)
        points = positions.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create line collection with color gradient
        lc = LineCollection(
            segments,
            cmap=plt.get_cmap('viridis'),
            norm=plt.Normalize(0, len(trajectory))
        )
        lc.set_array(np.arange(len(trajectory)))
        lc.set_linewidth(2)
        ax.add_collection(lc)

        # Mark start and end positions
        ax.plot(positions[0, 0], positions[0, 1], 'bo', markersize=10, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def create_comparison_plot(
    metrics_list: List[Dict],
    labels: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Create comparison bar charts for multiple policies.

    Args:
        metrics_list: List of metrics dictionaries from evaluate_random_policy
        labels: List of labels for each policy
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not available")
        return None

    if len(metrics_list) != len(labels):
        raise ValueError("metrics_list and labels must have same length")

    # Extract metrics
    success_rates = [m.get('success_rate', 0) for m in metrics_list]
    collision_rates = [m.get('collision_rate', 0) for m in metrics_list]
    avg_rewards = [m.get('avg_reward', 0) for m in metrics_list]
    avg_lengths = [m.get('avg_episode_length', 0) for m in metrics_list]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate
    axes[0, 0].bar(labels, success_rates, color='green', alpha=0.7)
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Collision rate
    axes[0, 1].bar(labels, collision_rates, color='red', alpha=0.7)
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Collision Rate Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Average reward
    axes[1, 0].bar(labels, avg_rewards, color='blue', alpha=0.7)
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].set_title('Average Reward Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Average episode length
    axes[1, 1].bar(labels, avg_lengths, color='orange', alpha=0.7)
    axes[1, 1].set_ylabel('Average Episode Length')
    axes[1, 1].set_title('Episode Length Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # For now, just show available functions
    print("\n" + "=" * 70)
    print("VISUALIZATION TOOLS FOR GRID NAVIGATION")
    print("=" * 70)
    print("\nAvailable functions:")
    print("  - create_trajectory_plot(): Plot robot trajectory with obstacles")
    print("  - create_comparison_plot(): Compare multiple policies")
    print("\nExample usage:")
    print("  from visualize import create_trajectory_plot")
    print("  create_trajectory_plot(trajectory, obstacles, goal, save_path='plot.png')")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
