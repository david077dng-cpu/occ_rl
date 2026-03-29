"""
Quick test script for the occupancy grid environment.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from occupancy_grid_env import OccupancyGridEnv
from grid_world import GridWorld
from robot_kinematics import HolonomicKinematics, RobotConfig


def test_grid_world():
    """Test the grid world simulation."""
    print("=" * 50)
    print("Testing GridWorld...")
    print("=" * 50)

    # Create grid world
    world = GridWorld(
        width=10.0,
        height=10.0,
        grid_resolution=0.3125,
        num_static_obstacles=5,
        num_dynamic_obstacles=2,
        random_seed=42,
    )

    print(f"Grid size: {world.grid_height} x {world.grid_width}")
    print(f"Static obstacles: {len(world.static_obstacles)}")
    print(f"Dynamic obstacles: {len(world.dynamic_obstacles)}")

    # Sample a position
    pos = world.sample_free_position()
    print(f"Sampled position: {pos}")

    # Check collision
    collision = world.check_collision(pos, robot_radius=0.3)
    print(f"Collision at sampled position: {collision}")

    # Get local occupancy grid
    local_grid = world.get_occupancy_grid_at_position(pos, robot_radius=0.3)
    print(f"Local grid shape: {local_grid.shape}")
    print(f"Occupied cells in local grid: {np.sum(local_grid > 0.5)}")

    print("✓ GridWorld test passed!")
    print()


def test_kinematics():
    """Test the robot kinematics."""
    print("=" * 50)
    print("Testing HolonomicKinematics...")
    print("=" * 50)

    # Create kinematics
    config = RobotConfig()
    kin = HolonomicKinematics(config)

    print(f"Wheel radius: {config.wheel_radius} m")
    print(f"Base radius: {config.base_radius} m")
    print(f"Wheel angles: {config.wheel_angles_deg} degrees")

    # Test body to wheels
    body_vel = np.array([0.5, 0.0, 0.0])  # Forward at 0.5 m/s
    wheel_vels = kin.body_to_wheels(body_vel)
    print(f"\nBody velocity: {body_vel}")
    print(f"Wheel velocities: {wheel_vels}")

    # Test wheels to body (inverse)
    body_vel_recovered = kin.wheels_to_body(wheel_vels)
    print(f"Recovered body velocity: {body_vel_recovered}")
    print(f"Reconstruction error: {np.linalg.norm(body_vel - body_vel_recovered)}")

    # Test state integration
    state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, vx, vy, omega]
    action = np.array([0.3, 0.1, 15.0])  # [vx, vy, omega_deg]

    new_state = kin.integrate_state(state, action * np.array([1, 1, np.pi/180]), 0.1)
    print(f"\nState integration:")
    print(f"Initial state: {state}")
    print(f"Action: {action}")
    print(f"New state: {new_state}")

    print("✓ HolonomicKinematics test passed!")
    print()


def test_environment():
    """Test the Gymnasium environment."""
    print("=" * 50)
    print("Testing OccupancyGridEnv...")
    print("=" * 50)

    # Create environment
    env = OccupancyGridEnv(
        world_width=10.0,
        world_height=10.0,
        max_episode_steps=100,
        random_seed=42,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"Occupancy grid shape: {obs['occupancy_grid'].shape}")
    print(f"Occupancy grid range: [{obs['occupancy_grid'].min():.2f}, {obs['occupancy_grid'].max():.2f}]")
    print(f"Robot pose: {obs['robot_pose']}")
    print(f"Target relative: {obs['target_relative']}")
    print(f"Velocity: {obs['velocity']}")
    print()

    print(f"Info: {info}")
    print()

    # Run a few random steps
    print("Running 10 random steps...")
    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"  Step {step + 1}: reward={reward:.3f}, "
              f"dist_to_goal={info['distance_to_goal']:.2f}, "
              f"collision={info['collision']}")

        if terminated or truncated:
            print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
            break

    print(f"Total reward: {total_reward:.3f}")

    print("✓ OccupancyGridEnv test passed!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("OCCUPANCY GRID RL ENVIRONMENT TEST SUITE")
    print("=" * 70 + "\n")

    try:
        test_grid_world()
        test_kinematics()
        test_environment()

        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
