"""
Policy Evaluation and Visualization

This module contains evaluation scripts and visualization tools for trained policies.
"""

from .eval_policy import (
    EpisodeResult,
    EvaluationMetrics,
    evaluate_random_policy,
    main as eval_main,
)

try:
    from .visualize import (
        create_trajectory_plot,
        create_comparison_plot,
    )
    __all__ = [
        "EpisodeResult",
        "EvaluationMetrics",
        "evaluate_random_policy",
        "eval_main",
        "create_trajectory_plot",
        "create_comparison_plot",
    ]
except ImportError:
    __all__ = [
        "EpisodeResult",
        "EvaluationMetrics",
        "evaluate_random_policy",
        "eval_main",
    ]
