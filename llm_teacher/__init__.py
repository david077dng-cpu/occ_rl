"""LLM Teacher module for imitation learning.

This module provides the LLMTeacher class that gets actions from an LLM
based on the current observation, for collecting demonstration data.
"""

from .llm_teacher import LLMTeacher

__all__ = ["LLMTeacher"]
