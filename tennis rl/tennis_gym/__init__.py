""" Registers the gym environments and exports the `gym.make` function.
"""

# Silencing pygame:
import os
import tennis_gym.agents

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Importing CLI:
from tennis_gym.cli import get_args

# Importing envs:
from tennis_gym.envs.tennis_env_simple import TennisEnvSimple

# Exporting make and register of envs:
from gymnasium import make, register

register(
    id="Tennis",
    entry_point="tennis_gym:TennisEnvSimple",
)

# Main names:
__all__ = [
    make.__name__,
    TennisEnvSimple.__name__,
]
