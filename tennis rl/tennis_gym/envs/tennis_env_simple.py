""" Implementation of a Flappy Bird OpenAI Gym environment that yields simple
numerical information about the game's state as observations.
"""

import numpy as np
import gymnasium
import pygame

from numpy import ndarray

from tennis_gym.envs.renderer import TennisRenderer
from tennis_gym.envs.game_logic import TennisLogic
from typing import Tuple, Any


class TennisEnvSimple(gymnasium.Env):
    def __init__(self,
                 screen_size: Tuple[int, int] = (500, 250),
                 paddle_size: int = 20,
                 ball_radius: int = 2,
                 max_score: int = 10,
                 FPS: int = 140) -> None:
        self.action_space = gymnasium.spaces.Discrete(3)
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float64
        )
        self._screen_size = screen_size
        self._paddle_size = paddle_size
        self._ball_radius = ball_radius
        self._max_score = max_score
        self._FPS = FPS

        self._renderer = None
        self._game = None

    def _get_observation(self):
        return np.array([self._game.ball.x,
                         self._game.ball.y,
                         self._game.ball.vel_x,
                         self._game.ball.vel_y,
                         self._game.player1.x,
                         self._game.player1.y,
                         self._game.player2.x,
                         self._game.player2.y], dtype=np.float64)

    def _check_done(self):
        return self._game.score1 >= self._max_score or self._game.score2 >= self._max_score

    def step(self,
             action: int,
    ) -> tuple[ndarray, int, bool, bool, dict[str, Any]]:
        scored, hit = self._game.update_state(action)
        reward = scored * 5 + hit
        terminated = self._check_done()
        obs = self._get_observation()
        truncated = False
        info = {"score1": self._game.score1,
                "score2": self._game.score2}
        return obs, reward, terminated, truncated, info

    def reset(self,
              seed: int = None,
              options: dict[str, Any] = None):
        """ Resets the environment (starts a new game). """
        super().reset(seed=seed)

        self._game = TennisLogic(screen_size=self._screen_size,
                                 paddle_size=self._paddle_size,
                                 ball_radius=self._ball_radius)
        if self._renderer is not None:
            self._renderer.game = self._game

        info = {"score1": self._game.score1,
                "score2": self._game.score2}
        return self._get_observation(), info

    def render(self) -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = TennisRenderer(screen_size=self._screen_size,
                                            paddle_size=self._paddle_size,
                                            ball_radius=self._ball_radius,
                                            FPS=self._FPS)
            self._renderer.game = self._game
            self._renderer.make_display()
        pygame.event.pump()
        self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        if self._renderer is not None:
            pygame.display.quit()
            pygame.quit()
            self._renderer = None
        super().close()
