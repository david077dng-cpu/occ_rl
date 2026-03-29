"""
Holonomic Robot Kinematics for 3-Omniwheel Robot (LeKiwi-style)

This module implements the kinematics for a holonomic mobile robot with
3 omni wheels arranged at 120° intervals (Kiwi drive configuration).

Key parameters (based on LeKiwi hardware):
- Wheel radius: 0.05m
- Base radius: 0.125m (center of robot to wheel contact point)
- Wheel angles: 0°, 120°, 240° (relative to robot forward)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RobotConfig:
    """Configuration for holonomic robot kinematics."""
    wheel_radius: float = 0.05      # meters
    base_radius: float = 0.125      # meters (robot center to wheel)
    wheel_angles_deg: Tuple[float, float, float] = (0.0, 120.0, 240.0)

    @property
    def wheel_angles_rad(self) -> Tuple[float, float, float]:
        """Wheel angles in radians."""
        return tuple(np.deg2rad(a) for a in self.wheel_angles_deg)


class HolonomicKinematics:
    """
    Kinematics for 3-wheel holonomic (Kiwi) drive robot.

    The forward kinematics maps wheel velocities to body velocities:
        v_body = J * ω_wheels

    where J is the Jacobian matrix that depends on wheel geometry.

    The inverse kinematics maps body velocities to wheel velocities:
        ω_wheels = J^(-1) * v_body

    Body velocity vector: [vx, vy, ω] in robot frame
        - vx: forward velocity (m/s)
        - vy: lateral velocity (m/s)
        - ω: angular velocity (rad/s)

    Wheel velocity vector: [ω1, ω2, ω3] in rad/s
    """

    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Initialize kinematics with given configuration.

        Args:
            config: Robot configuration. Uses default LeKiwi config if None.
        """
        self.config = config or RobotConfig()
        self._compute_jacobian()

    def _compute_jacobian(self):
        """
        Compute the forward and inverse Jacobian matrices.

        For a wheel at angle β from robot forward direction:
            vx_wheel = -vx * sin(β) + vy * cos(β) + ω * R

        where R is base radius.

        The wheel angular velocity is:
            ω_wheel = vx_wheel / r

        where r is wheel radius.

        This gives the inverse Jacobian J_inv such that:
            ω_wheels = J_inv @ [vx, vy, ω]^T
        """
        r = self.config.wheel_radius
        R = self.config.base_radius
        angles = self.config.wheel_angles_rad

        # Inverse Jacobian: maps body velocities to wheel velocities
        # Each row corresponds to one wheel
        self.J_inv = np.zeros((3, 3))

        for i, beta in enumerate(angles):
            # For wheel at angle beta:
            # vx_wheel = -vx * sin(beta) + vy * cos(beta) + omega * R
            self.J_inv[i, 0] = -np.sin(beta) / r  # vx contribution
            self.J_inv[i, 1] = np.cos(beta) / r   # vy contribution
            self.J_inv[i, 2] = R / r              # omega contribution

        # Forward Jacobian: maps wheel velocities to body velocities
        self.J = np.linalg.pinv(self.J_inv)

    def body_to_wheels(self, body_velocity: np.ndarray) -> np.ndarray:
        """
        Convert body velocities to wheel velocities.

        Args:
            body_velocity: [vx, vy, omega] in m/s, m/s, rad/s

        Returns:
            wheel_velocities: [omega1, omega2, omega3] in rad/s
        """
        return self.J_inv @ body_velocity

    def wheels_to_body(self, wheel_velocities: np.ndarray) -> np.ndarray:
        """
        Convert wheel velocities to body velocities.

        Args:
            wheel_velocities: [omega1, omega2, omega3] in rad/s

        Returns:
            body_velocity: [vx, vy, omega] in m/s, m/s, rad/s
        """
        return self.J @ wheel_velocities

    def integrate_state(
        self,
        state: np.ndarray,
        body_velocity: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Integrate robot state forward by time step dt.

        State: [x, y, theta, vx, vy, omega]
            - x, y: world position (m)
            - theta: heading angle (rad)
            - vx, vy: body velocities (m/s)
            - omega: angular velocity (rad/s)

        Args:
            state: Current state [6,]
            body_velocity: Body velocity commands [vx, vy, omega]
            dt: Time step (s)

        Returns:
            new_state: Updated state [6,]
        """
        x, y, theta = state[0], state[1], state[2]

        # Body velocities
        vx_body, vy_body, omega = body_velocity

        # Convert body velocity to world frame
        vx_world = vx_body * np.cos(theta) - vy_body * np.sin(theta)
        vy_world = vx_body * np.sin(theta) + vy_body * np.cos(theta)

        # Integrate using Euler method
        new_x = x + vx_world * dt
        new_y = y + vy_world * dt
        new_theta = theta + omega * dt

        # Normalize angle to [-pi, pi]
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        return np.array([
            new_x, new_y, new_theta,
            vx_body, vy_body, omega
        ])

    def compute_wheel_speeds_from_action(
        self,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Convert action (velocity commands) to wheel speeds.

        This is the main interface for RL environment.
        Action: [vx, vy, omega] where
            - vx: forward velocity (m/s)
            - vy: lateral velocity (m/s)
            - omega: angular velocity (deg/s, will be converted to rad/s)

        Args:
            action: [vx, vy, omega_deg]

        Returns:
            wheel_speeds: [omega1, omega2, omega3] in rad/s
        """
        # Convert omega from deg/s to rad/s
        body_velocity = np.array([
            action[0],
            action[1],
            np.deg2rad(action[2])
        ])

        return self.body_to_wheels(body_velocity)
