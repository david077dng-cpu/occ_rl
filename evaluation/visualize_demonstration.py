"""
Visualize LLM Demonstration Episodes

This script loads a collected demonstration dataset and visualizes selected
episodes by plotting the robot trajectory on the occupancy grid.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def visualize_episode(
    dataset_path: str,
    episode_idx: int,
    output_path: str = None,
    show: bool = True,
):
    """
    Visualize a single demonstration episode from the dataset.

    Args:
        dataset_path: Path to .npz dataset
        episode_idx: Which episode to visualize
        output_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)

    # Extract all data
    obs_grid = data['obs_grid']
    obs_robot_pose = data['obs_robot_pose']
    obs_target_relative = data['obs_target_relative']
    actions = data['actions']
    dones = data['dones']

    # Find the start and end indices of the requested episode
    # We need to scan through dones to find where the episode starts and ends
    episode_starts = [0]
    for i, done in enumerate(dones):
        if done == 1.0 and i < len(dones) - 1:
            episode_starts.append(i + 1)

    if episode_idx >= len(episode_starts):
        raise ValueError(f"Episode {episode_idx} not found, only {len(episode_starts)} episodes in dataset")

    start_idx = episode_starts[episode_idx]
    if episode_idx == len(episode_starts) - 1:
        end_idx = len(obs_grid)
    else:
        end_idx = episode_starts[episode_idx + 1]

    # Extract this episode's data
    episode_grid = obs_grid[start_idx:end_idx]
    episode_poses = obs_robot_pose[start_idx:end_idx]
    episode_actions = actions[start_idx:end_idx]

    print(f"Visualizing episode {episode_idx}:")
    print(f"  Steps: {len(episode_grid)}")
    print(f"  Start position: {episode_poses[0][:2]}")
    print(f"  End position: {episode_poses[-1][:2]}")

    # Get the full world grid (we need to reconstruct it from the local observations)
    # Actually we just plot the trajectory on a blank canvas and show obstacles
    # that are in the local view when the robot is there

    world_size = 10.0
    grid_resolution = world_size / 32 * 2  # 2x2 blocks per cell in original 32x32

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    # Set axis limits
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'LLM Teacher Demonstration - Episode {episode_idx}\n{len(episode_poses)} steps', fontsize=14)

    # Plot grid lines
    ax.grid(True, alpha=0.3)

    # Extract trajectory
    xs = [pose[0] for pose in episode_poses]
    ys = [pose[1] for pose in episode_poses]

    # Plot trajectory
    ax.plot(xs, ys, 'b-', linewidth=2, label='Robot trajectory', zorder=3)

    # Mark start and end
    ax.scatter(xs[0], ys[0], color='green', s=200, zorder=5, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red', s=200, zorder=5, label='Goal reached' if dones[end_idx-1] == 1 else 'Timeout')

    # Plot obstacles that the robot encountered
    # For each step, add obstacles from the local grid to world coordinates
    for step in range(len(episode_grid)):
        robot_x, robot_y, _ = episode_poses[step]
        local_grid = episode_grid[step]  # 32x32

        # Convert local grid coordinates to world coordinates
        # Local grid is centered at robot position, 10m world -> 32 cells = 0.3125m per cell
        cell_size = 10.0 / 32

        for ly in range(32):
            for lx in range(32):
                if local_grid[ly, lx] > 0.5:
                    # Calculate world coordinates for this obstacle cell
                    world_x = robot_x - 5.0 + (lx + 0.5) * cell_size
                    world_y = robot_y - 5.0 + (ly + 0.5) * cell_size

                    # Draw obstacle rectangle
                    rect = Rectangle(
                        (world_x - cell_size/2, world_y - cell_size/2),
                        cell_size, cell_size,
                        color='darkred',
                        alpha=0.7,
                        zorder=2
                    )
                    ax.add_patch(rect)

    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()

    return fig, ax


def visualize_multiple(
    dataset_path: str,
    num_episodes: int = 4,
    output_dir: str = './visualizations',
):
    """Visualize multiple episodes and save figures."""
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    metadata = data['metadata'].item()
    print(f"\nDataset metadata:")
    print(f"  Total episodes: {metadata['num_episodes']}")
    print(f"  Success rate: {metadata['success_rate']:.2%}")
    print(f"  Average length: {metadata['avg_episode_length']:.1f}")

    num_to_visualize = min(num_episodes, metadata['num_episodes'])

    for i in range(num_to_visualize):
        output_path = os.path.join(output_dir, f'episode_{i:03d}.png')
        visualize_episode(dataset_path, i, output_path, show=False)
        print(f"Saved: {output_path}")

    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize LLM demonstration episodes')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to demonstration dataset .npz')
    parser.add_argument('--episode', type=int, default=0,
                        help='Episode index to visualize')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure')
    parser.add_argument('--multiple', action='store_true',
                        help='Visualize multiple episodes and save all')
    parser.add_argument('--num-multiple', type=int, default=4,
                        help='Number of episodes to visualize if --multiple is set')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                        help='Output directory for multiple visualizations')

    args = parser.parse_args()

    if args.multiple:
        visualize_multiple(args.dataset, args.num_multiple, args.output_dir)
    else:
        visualize_episode(args.dataset, args.episode, args.output, show=True)


if __name__ == '__main__':
    main()
