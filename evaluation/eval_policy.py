"""
Policy Evaluation Script for Grid Navigation

This script evaluates trained policies and computes metrics:
- Success rate (reaching goal)
- Average episode length
- Collision rate
- Average reward
- Distance traveled

Usage:
    # Evaluate a checkpoint
    python eval_policy.py /path/to/checkpoint \
        --num-episodes=100 \
        --render

    # Save trajectories
    python eval_policy.py /path/to/checkpoint \
        --num-episodes=50 \
        --save-trajectories=./trajectories.npz
"""

import argparse
import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    episode_id: int
    success: bool
    collision: bool
    episode_length: int
    total_reward: float
    initial_distance: float
    final_distance: float
    distance_traveled: float
    time_to_goal: Optional[float] = None
    trajectory: Optional[np.ndarray] = None  # Shape: (T, 3) for [x, y, theta]

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding trajectory for JSON serialization)."""
        result = {
            "episode_id": self.episode_id,
            "success": self.success,
            "collision": self.collision,
            "episode_length": self.episode_length,
            "total_reward": float(self.total_reward),
            "initial_distance": float(self.initial_distance),
            "final_distance": float(self.final_distance),
            "distance_traveled": float(self.distance_traveled),
            "time_to_goal": float(self.time_to_goal) if self.time_to_goal else None,
        }
        return result


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from multiple evaluation episodes."""
    num_episodes: int
    success_rate: float
    collision_rate: float
    avg_episode_length: float
    avg_reward: float
    avg_distance_traveled: float
    avg_time_to_goal: Optional[float]
    std_reward: float
    min_reward: float
    max_reward: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_episodes": self.num_episodes,
            "success_rate": float(self.success_rate),
            "collision_rate": float(self.collision_rate),
            "avg_episode_length": float(self.avg_episode_length),
            "avg_reward": float(self.avg_reward),
            "avg_distance_traveled": float(self.avg_distance_traveled),
            "avg_time_to_goal": float(self.avg_time_to_goal) if self.avg_time_to_goal else None,
            "std_reward": float(self.std_reward),
            "min_reward": float(self.min_reward),
            "max_reward": float(self.max_reward),
        }

    def print_summary(self):
        """Print formatted summary of metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Number of Episodes:     {self.num_episodes}")
        print(f"Success Rate:           {self.success_rate * 100:.1f}%")
        print(f"Collision Rate:         {self.collision_rate * 100:.1f}%")
        print(f"Avg Episode Length:     {self.avg_episode_length:.1f} steps")
        print(f"Avg Reward:             {self.avg_reward:.2f} ± {self.std_reward:.2f}")
        print(f"Reward Range:           [{self.min_reward:.2f}, {self.max_reward:.2f}]")
        print(f"Avg Distance Traveled:  {self.avg_distance_traveled:.2f} m")
        if self.avg_time_to_goal:
            print(f"Avg Time to Goal:       {self.avg_time_to_goal:.2f} s")
        print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained grid navigation policy"
    )

    # Checkpoint
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default=None,
        help="Path to checkpoint directory or file"
    )

    # Evaluation settings
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Environment settings
    parser.add_argument(
        "--world-size",
        type=float,
        default=10.0,
        help="World size in meters"
    )
    parser.add_argument(
        "--num-static-obstacles",
        type=int,
        default=5,
        help="Number of static obstacles"
    )
    parser.add_argument(
        "--num-dynamic-obstacles",
        type=int,
        default=2,
        help="Number of dynamic obstacles"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-trajectories",
        type=str,
        default=None,
        help="Save trajectories to NPZ file"
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default=None,
        help="Save metrics to JSON file"
    )

    # Rendering
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (saves images)"
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=1,
        help="Render every N episodes"
    )
    parser.add_argument(
        "--render-output",
        type=str,
        default=None,
        help="Directory to save rendered frames"
    )

    # Other options
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no sampling)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def create_env_config(args) -> Dict:
    """Create environment configuration."""
    return {
        "world_width": args.world_size,
        "world_height": args.world_size,
        "grid_resolution": 0.3125,
        "num_static_obstacles": args.num_static_obstacles,
        "num_dynamic_obstacles": args.num_dynamic_obstacles,
        "max_episode_steps": args.max_episode_steps,
        "goal_threshold": 0.5,
        "collision_threshold": 0.3,
        "random_seed": args.seed,
        "render_mode": None,  # We'll handle rendering separately
    }


def evaluate_random_policy(
    env,
    num_episodes: int,
    max_steps: int,
    seed: int = 42
) -> Tuple[List[EpisodeResult], EvaluationMetrics]:
    """
    Evaluate using random policy as a baseline.

    This is used when no checkpoint is provided for quick testing.
    """
    np.random.seed(seed)

    episodes = []
    all_rewards = []

    for episode_id in range(num_episodes):
        obs, info = env.reset(seed=seed + episode_id)
        episode_reward = 0.0
        episode_length = 0
        collision = False
        goal_reached = False
        trajectory = []

        initial_distance = info.get("distance_to_goal", 0.0)

        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Track trajectory
            robot_pose = obs.get("robot_pose", None)
            if robot_pose is not None:
                trajectory.append(robot_pose.copy())

            # Check termination
            if info.get("collision", False):
                collision = True

            if info.get("goal_reached", False):
                goal_reached = True

            if terminated or truncated:
                break

        final_distance = info.get("distance_to_goal", initial_distance)
        distance_traveled = sum(
            np.linalg.norm(trajectory[i+1][:2] - trajectory[i][:2])
            for i in range(len(trajectory) - 1)
        ) if len(trajectory) > 1 else 0.0

        episode_result = EpisodeResult(
            episode_id=episode_id,
            success=goal_reached,
            collision=collision,
            episode_length=episode_length,
            total_reward=episode_reward,
            initial_distance=initial_distance,
            final_distance=final_distance,
            distance_traveled=distance_traveled,
            time_to_goal=episode_length * 0.1 if goal_reached else None,
            trajectory=np.array(trajectory) if trajectory else None,
        )

        episodes.append(episode_result)
        all_rewards.append(episode_reward)

    # Compute aggregate metrics
    rewards = np.array(all_rewards)
    success_count = sum(1 for e in episodes if e.success)
    collision_count = sum(1 for e in episodes if e.collision)
    successful_episodes = [e for e in episodes if e.success]

    metrics = EvaluationMetrics(
        num_episodes=num_episodes,
        success_rate=success_count / num_episodes,
        collision_rate=collision_count / num_episodes,
        avg_episode_length=np.mean([e.episode_length for e in episodes]),
        avg_reward=np.mean(rewards),
        avg_distance_traveled=np.mean([e.distance_traveled for e in episodes]),
        avg_time_to_goal=np.mean([e.time_to_goal for e in successful_episodes]) if successful_episodes else None,
        std_reward=np.std(rewards),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
    )

    return episodes, metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create environment config
    env_config = create_env_config(args)

    # Import environment
    from occupancy_grid_rl.envs import OccupancyGridEnv

    # Create environment
    env = OccupancyGridEnv(**env_config)

    print(f"\n{'='*70}")
    print("POLICY EVALUATION")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint or 'None (random policy)'}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_episode_steps}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

    # Run evaluation
    start_time = time.time()

    if args.checkpoint:
        # TODO: Load checkpoint and evaluate trained policy
        # For now, use random policy
        print("Note: Trained policy evaluation not yet implemented.")
        print("Using random policy for demonstration.\n")
        episodes, metrics = evaluate_random_policy(
            env, args.num_episodes, args.max_episode_steps, args.seed
        )
    else:
        # Use random policy
        print("Evaluating random policy...\n")
        episodes, metrics = evaluate_random_policy(
            env, args.num_episodes, args.max_episode_steps, args.seed
        )

    elapsed_time = time.time() - start_time

    # Print summary
    metrics.print_summary()

    print(f"\nEvaluation completed in {elapsed_time:.1f} seconds")

    # Save results
    if args.save_metrics:
        metrics_path = args.save_metrics
    else:
        metrics_path = os.path.join(args.output_dir, "metrics.json")

    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save trajectories
    if args.save_trajectories:
        trajectories = {}
        for i, episode in enumerate(episodes):
            if episode.trajectory is not None:
                trajectories[f"episode_{i}"] = episode.trajectory

        if trajectories:
            np.savez_compressed(args.save_trajectories, **trajectories)
            print(f"Trajectories saved to: {args.save_trajectories}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
