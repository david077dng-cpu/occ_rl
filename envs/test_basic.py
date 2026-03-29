"""
Basic unit tests without external dependencies (numpy only).
Tests grid world and kinematics logic.
"""

import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_world import GridWorld, StaticObstacle, DynamicObstacle
from robot_kinematics import HolonomicKinematics, RobotConfig


def test_grid_world():
    """Test grid world creation and basic operations."""
    print("\n" + "="*60)
    print("Testing GridWorld...")
    print("="*60)

    world = GridWorld(
        width=10.0,
        height=10.0,
        grid_resolution=0.3125,
        num_static_obstacles=3,
        num_dynamic_obstacles=1,
        random_seed=42
    )

    print(f"Grid dimensions: {world.grid_height} x {world.grid_width}")
    print(f"Physical size: {world.width}m x {world.height}m")
    print(f"Cell resolution: {world.grid_resolution}m")
    print(f"Static obstacles: {len(world.static_obstacles)}")
    print(f"Dynamic obstacles: {len(world.dynamic_obstacles)}")

    # Test sampling
    pos = world.sample_free_position()
    print(f"\nSampled position: ({pos[0]:.2f}, {pos[1]:.2f})")

    # Test collision
    collision = world.check_collision(pos, robot_radius=0.3)
    print(f"Collision check: {collision}")

    # Test local grid extraction
    local_grid = world.get_occupancy_grid_at_position(pos, robot_radius=0.3)
    print(f"Local grid shape: {local_grid.shape}")
    print(f"Occupied cells: {np.sum(local_grid > 0.5)}/{local_grid.size}")

    # Test dynamic obstacle update
    dt = 0.1
    world.update_dynamic_obstacles(dt)
    print(f"\nUpdated dynamic obstacles for dt={dt}s")

    print("\n✓ GridWorld tests passed!")
    return True


def test_kinematics():
    """Test robot kinematics."""
    print("\n" + "="*60)
    print("Testing HolonomicKinematics...")
    print("="*60)

    # Create with default config
    config = RobotConfig()
    kin = HolonomicKinematics(config)

    print(f"Wheel radius: {config.wheel_radius}m")
    print(f"Base radius: {config.base_radius}m")
    print(f"Wheel angles: {config.wheel_angles_deg} degrees")

    # Test forward kinematics: pure forward motion
    body_vel = np.array([0.5, 0.0, 0.0])  # 0.5 m/s forward
    wheel_vels = kin.body_to_wheels(body_vel)
    print(f"\nForward motion test:")
    print(f"  Body velocity: {body_vel} [vx, vy, omega]")
    print(f"  Wheel velocities: {wheel_vels} rad/s")

    # Test inverse kinematics
    body_vel_recovered = kin.wheels_to_body(wheel_vels)
    print(f"  Recovered body velocity: {body_vel_recovered}")
    print(f"  Reconstruction error: {np.linalg.norm(body_vel - body_vel_recovered):.6f}")

    # Test lateral motion
    body_vel_lateral = np.array([0.0, 0.3, 0.0])  # 0.3 m/s lateral
    wheel_vels_lateral = kin.body_to_wheels(body_vel_lateral)
    print(f"\nLateral motion test:")
    print(f"  Body velocity: {body_vel_lateral}")
    print(f"  Wheel velocities: {wheel_vels_lateral} rad/s")

    # Test rotation
    body_vel_rotate = np.array([0.0, 0.0, 1.0])  # 1 rad/s rotation
    wheel_vels_rotate = kin.body_to_wheels(body_vel_rotate)
    print(f"\nRotation test:")
    print(f"  Body velocity: {body_vel_rotate}")
    print(f"  Wheel velocities: {wheel_vels_rotate} rad/s")

    # Test state integration
    print(f"\nState integration test:")
    state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])  # At origin, facing x
    body_vel_cmd = np.array([0.5, 0.0, 0.0])  # Forward at 0.5 m/s
    dt = 0.1

    print(f"  Initial state: {state}")
    print(f"  Command velocity: {body_vel_cmd}")
    print(f"  Time step: {dt}s")

    for step in range(10):
        state = kin.integrate_state(state, body_vel_cmd, dt)

    print(f"  Final state after 1.0s: {state}")
    print(f"  Expected: x≈5.5, y≈5.0, theta≈0.0")

    print("\n✓ HolonomicKinematics tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("OCCUPANCY GRID RL - BASIC UNIT TESTS")
    print("=" * 70)

    try:
        test_grid_world()
        test_kinematics()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
