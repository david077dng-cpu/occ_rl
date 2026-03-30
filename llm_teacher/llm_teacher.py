"""
LLM Teacher for Imitation Learning

This module implements the LLMTeacher that observes the environment and
outputs expert actions (vx, vy, omega) for the robot. The LLM acts as a
teacher, providing demonstrations for behavior cloning.
"""

import os
import re
import json
import hashlib
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

# Load environment variables from .env if it exists
import dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)


@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "gpt-4o"
    cache_dir: str = "./data/llm_cache"
    temperature: float = 0.0


class LLMTeacher:
    """
    LLM Teacher that provides expert actions given observations.

    Converts the observation (occupancy grid, robot pose, target position,
    velocity) into a text prompt, queries the LLM, and parses the action.
    """

    # Action bounds from environment
    VX_MIN = -0.5
    VX_MAX = 0.5
    VY_MIN = -0.5
    VY_MAX = 0.5
    OMEGA_MIN = -90.0
    OMEGA_MAX = 90.0

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLMTeacher.

        Args:
            config: LLM configuration. If not provided, reads from env vars:
                - ANTHROPIC_AUTH_TOKEN (or LLM_API_KEY)
                - ANTHROPIC_BASE_URL (or LLM_BASE_URL)
                - ANTHROPIC_MODEL (or LLM_MODEL_NAME)
                - LLM_CACHE_DIR (default: ./data/llm_cache)
        """
        # Get config from environment if not provided
        # Follow existing Daily Observation pattern from this project
        if config is None:
            api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("LLM_API_KEY")
            base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("LLM_BASE_URL")
            model_name = os.environ.get("ANTHROPIC_MODEL") or os.environ.get("LLM_MODEL_NAME", "ark-code-latest")
            # Strip trailing slash to avoid double slashes in URL
            if base_url:
                base_url = base_url.rstrip('/')
            config = LLMConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                cache_dir=os.environ.get("LLM_CACHE_DIR", "./data/llm_cache"),
            )

        self.config = config

        # Initialize OpenAI client
        client_kwargs = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)

        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)

        # System prompt that describes the task
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt describing the task."""
        return """You are an expert robot navigation controller.
Your job is to output velocity commands to guide a holonomic robot
to a target goal while avoiding obstacles in a 2D grid world.

The robot can move in any direction (holonomic) with velocities:
- vx: forward/backward velocity [-0.5, 0.5] m/s
- vy: left/right velocity [-0.5, 0.5] m/s
- omega: rotational velocity [-90, 90] deg/s

OUTPUT FORMAT (strictly follow this):
vx: <value>, vy: <value>, omega: <value>

Just output the three values - no other explanation needed.
"""

    def _grid_to_ascii(self, occupancy_grid: np.ndarray) -> str:
        """
        Convert 32x32 occupancy grid to ASCII visualization.

        Robot is at center (16, 16), marked with 'R'.
        Goal direction is marked with 'G'.
        Obstacles are 'X', free space is '.'.

        For display purposes, we downsample to 16x16 to keep it compact.
        """
        # 32x32 -> 16x16 by 2x2 pooling
        ascii_grid = []
        for y in range(0, 32, 2):
            line = []
            for x in range(0, 32, 2):
                # Check if any obstacle in 2x2 block
                block = occupancy_grid[y:y+2, x:x+2]
                if block.mean() > 0.5:
                    char = 'X'
                else:
                    char = '.'
                line.append(char)
            ascii_grid.append(''.join(line))

        # Mark robot at center (8, 8) in 16x16
        grid_lines = [list(line) for line in ascii_grid]
        grid_lines[8][8] = 'R'

        return '\n'.join(''.join(line) for line in grid_lines)

    def _get_cache_key(self, obs: Dict[str, np.ndarray]) -> str:
        """Generate a cache key from observation."""
        # Hash the observation to get a unique key
        obs_bytes = b''
        for k in sorted(obs.keys()):
            obs_bytes += obs[k].tobytes()
        return hashlib.md5(obs_bytes).hexdigest()[:16]

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if it exists."""
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data.get("response")
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache the response to disk."""
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump({"cache_key": cache_key, "response": response}, f)

    def format_prompt(self, obs: Dict[str, np.ndarray]) -> str:
        """
        Format observation into a text prompt for the LLM.

        Args:
            obs: Observation dictionary with:
                - occupancy_grid: (32, 32) np.ndarray
                - robot_pose: (3,) [x, y, theta] world pose
                - target_relative: (2,) [dx, dy] normalized relative target
                - velocity: (3,) [vx, vy, omega] current velocity

        Returns:
            Formatted prompt string.
        """
        grid = obs['occupancy_grid']
        robot_pose = obs['robot_pose']
        target_relative = obs['target_relative']
        velocity = obs['velocity']

        # Convert grid to ASCII
        grid_ascii = self._grid_to_ascii(grid)

        # Denormalize target relative - it's normalized to [-1, 1] in world size 10m
        dx = target_relative[0] * 5.0
        dy = target_relative[1] * 5.0
        distance = np.sqrt(dx**2 + dy**2)

        prompt = f"""Current state:

GRID (R = robot, X = obstacle, . = free):
{grid_ascii}

Robot current position: x = {robot_pose[0]:.2f} m, y = {robot_pose[1]:.2f} m
Robot current heading: theta = {robot_pose[2]:.2f} rad
Target relative position: dx = {dx:.2f} m, dy = {dy:.2f} m
Distance to target: {distance:.2f} m
Current velocities: vx = {velocity[0]:.2f} m/s, vy = {velocity[1]:.2f} m/s, omega = {velocity[2]:.2f} deg/s

Output the optimal velocities vx, vy, omega to move toward the target while avoiding obstacles:
"""
        return prompt

    def _parse_response(self, response: str) -> Tuple[float, float, float]:
        """
        Parse LLM response to extract vx, vy, omega.

        Args:
            response: LLM response text.

        Returns:
            Tuple of (vx, vy, omega).

        Raises:
            ValueError: if parsing fails.
        """
        # Look for patterns like "vx: 0.3, vy: -0.1, omega: 0.0"
        pattern = r'vx:\s*([+-]?\d*\.?\d+)\s*[,;]?\s*vy:\s*([+-]?\d*\.?\d+)\s*[,;]?\s*omega:\s*([+-]?\d*\.?\d+)'
        match = re.search(pattern, response.lower())

        if not match:
            # Try alternative pattern without labels
            pattern2 = r'([+-]?\d*\.?\d+)\s*[, ]\s*([+-]?\d*\.?\d+)\s*[, ]\s*([+-]?\d*\.?\d+)'
            match = re.search(pattern2, response)
            if not match:
                raise ValueError(f"Could not parse action from response: {response}")
            vx, vy, omega = float(match.group(1)), float(match.group(2)), float(match.group(3))
        else:
            vx, vy, omega = float(match.group(1)), float(match.group(2)), float(match.group(3))

        # Clip to valid ranges
        vx = np.clip(vx, self.VX_MIN, self.VX_MAX)
        vy = np.clip(vy, self.VY_MIN, self.VY_MAX)
        omega = np.clip(omega, self.OMEGA_MIN, self.OMEGA_MAX)

        return vx, vy, omega

    def get_action(self, obs: Dict[str, np.ndarray], use_cache: bool = True) -> np.ndarray:
        """
        Get action from LLM teacher for current observation.

        Args:
            obs: Current observation dictionary.
            use_cache: Whether to use cached responses (default: True).

        Returns:
            action: np.ndarray of shape (3,) - [vx, vy, omega].
        """
        cache_key = self._get_cache_key(obs)

        # Check cache first
        if use_cache:
            cached = self._get_cached_response(cache_key)
            if cached is not None:
                vx, vy, omega = self._parse_response(cached)
                return np.array([vx, vy, omega], dtype=np.float32)

        # Format prompt
        prompt = self.format_prompt(obs)

        # Query LLM (OpenAI API format) with retries
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Retry logic for API errors (rate limits, network issues)
        max_retries = 8
        retry_delay = 10.0  # Longer delay for rate limits
        response_text = None

        for retry in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=64,
                )
                response_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"[LLM API] Attempt {retry+1}/{max_retries} failed: {type(e).__name__}: {e}")
                if retry < max_retries - 1:
                    import time
                    delay = retry_delay * (retry + 1)
                    print(f"  Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    raise

        assert response_text is not None, "All API attempts failed"

        # Cache the response
        if use_cache:
            self._cache_response(cache_key, response_text)

        # Parse action
        vx, vy, omega = self._parse_response(response_text)
        return np.array([vx, vy, omega], dtype=np.float32)
