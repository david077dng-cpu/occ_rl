"""
Common Utilities for Occupancy Grid RL

Shared utility functions used across the RL pipeline.
"""

import os
import json
import pickle
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    use_pickle: bool = True
) -> None:
    """
    Save checkpoint data to file.

    Args:
        data: Dictionary containing checkpoint data
        filepath: Path to save checkpoint
        use_pickle: If True, use pickle format, else JSON
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if use_pickle:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_data = convert_to_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)


def load_checkpoint(
    filepath: Union[str, Path],
    use_pickle: bool = True
) -> Dict[str, Any]:
    """
    Load checkpoint data from file.

    Args:
        filepath: Path to checkpoint file
        use_pickle: If True, use pickle format, else JSON

    Returns:
        Dictionary containing checkpoint data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    if use_pickle:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def compute_path_length(path: np.ndarray) -> float:
    """
    Compute total path length from sequence of positions.

    Args:
        path: Array of shape (N, 2+) containing positions

    Returns:
        Total path length
    """
    if len(path) < 2:
        return 0.0

    positions = path[:, :2]  # Extract x, y
    distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    return float(np.sum(distances))


def interpolate_path(
    path: np.ndarray,
    num_points: int
) -> np.ndarray:
    """
    Interpolate path to have specified number of points.

    Args:
        path: Array of shape (N, D) containing path points
        num_points: Number of points to interpolate to

    Returns:
        Interpolated path of shape (num_points, D)
    """
    if len(path) < 2:
        return np.tile(path[0], (num_points, 1)) if len(path) > 0 else np.zeros((num_points, path.shape[1]))

    # Compute cumulative distance along path
    positions = path[:, :2]
    distances = np.concatenate([[0], np.cumsum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))])

    if distances[-1] == 0:
        return np.tile(path[0], (num_points, 1))

    # Normalize distances to [0, 1]
    t = distances / distances[-1]

    # Interpolate at evenly spaced points
    t_new = np.linspace(0, 1, num_points)

    interpolated = np.zeros((num_points, path.shape[1]))
    for i in range(path.shape[1]):
        interpolated[:, i] = np.interp(t_new, t, path[:, i])

    return interpolated


class Logger:
    """Simple logger for tracking metrics."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            ensure_dir(self.log_dir)

        self.metrics = defaultdict(list)
        self.step = 0

    def log(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if step is None:
            step = self.step
        self.metrics[key].append((step, value))

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log(key, value, step)

    def increment_step(self):
        """Increment the global step counter."""
        self.step += 1

    def save(self, filename: Optional[str] = None):
        """Save logged metrics to file."""
        if not self.log_dir:
            return

        if filename is None:
            filename = "metrics.json"

        filepath = self.log_dir / filename

        # Convert to serializable format
        data = {
            "metrics": {k: [(int(s), float(v)) for s, v in vals]
                       for k, vals in self.metrics.items()},
            "step": self.step,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filename: Optional[str] = None):
        """Load logged metrics from file."""
        if not self.log_dir:
            return

        if filename is None:
            filename = "metrics.json"

        filepath = self.log_dir / filename

        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics = defaultdict(list, {k: [(int(s), float(v)) for s, v in vals]
                                         for k, vals in data.get("metrics", {}).items()})
        self.step = data.get("step", 0)


def main():
    """Main utility function."""
    print("\n" + "=" * 70)
    print("COMMON UTILITIES FOR OCCUPANCY GRID RL")
    print("=" * 70)
    print("\nAvailable utilities:")
    print("  - save_checkpoint(): Save model checkpoints")
    print("  - load_checkpoint(): Load model checkpoints")
    print("  - ensure_dir(): Create directories if needed")
    print("  - compute_path_length(): Calculate total path distance")
    print("  - interpolate_path(): Interpolate path to N points")
    print("  - normalize_angle(): Normalize angle to [-pi, pi]")
    print("  - Logger: Simple metrics logging utility")
    print("\nImport with: from occupancy_grid_rl.utils import *")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    from collections import defaultdict
    sys.exit(main())
