""" Different agents available for play. """
import numpy as np
import pygame
import torch
import torch.nn as nn


class Agent:
    """ Abstract class for agents. """
    def __init__(self,
                 side: int):
        pass

    def act(self,
            obs: np.ndarray
            ) -> int:
        pass


class HumanAgent(Agent):
    """ Human agent controlled by W and S. """
    def __init__(self,
                 side: int):
        self._side = side
        self._action = 0
        pass

    def act(self,
            event: list
            ) -> int:
        if self._side == 1:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self._action = 1
                if event.key == pygame.K_s:
                    self._action = 2

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self._action = 0
                if event.key == pygame.K_s:
                    self._action = 0

        elif self._side == 2:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._action = 1
                if event.key == pygame.K_DOWN:
                    self._action = 2

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self._action = 0
                if event.key == pygame.K_DOWN:
                    self._action = 0
        return self._action


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def predict(self, X):
        with torch.no_grad():
            result = self.net(X)
        return result


class DQNAgent(Agent):
    """ DQN agent. """
    def __init__(self,
                 side: int,
                 screen_width: int = 500):
        self._side = side
        self._screen_width = screen_width
        self._model = Model(8, 3)
        self._model.net.load_state_dict(torch.load('tennis_gym/models/32batch_10000memory_40000episodes'))

    def act(self,
            obs: np.ndarray
            ) -> int:
        if self._side == 2:
            obs[0] = self._screen_width - obs[0]
            obs[2] *= -1
            obs[5], obs[7] = obs[7], obs[5]
        obs = torch.Tensor(obs.reshape(1, len(obs)))
        act_values = self._model.predict(obs)
        return np.argmax(act_values[0])


class GodAgent(Agent):
    def __init__(self,
                 side: int,
                 paddle_size: int = 20):
        self._side = side
        self._paddle_size = paddle_size
        pass

    def act(self,
            obs: np.ndarray
            ) -> int:
        action = 0
        if self._side == 1:
            if obs[1] < obs[5] - self._paddle_size // 10:
                action = 1
            if obs[1] > obs[5] - self._paddle_size // 10:
                action = 2

        elif self._side == 2:
            if obs[1] < obs[7] - self._paddle_size // 10:
                action = 1
            if obs[1] > obs[7] - self._paddle_size // 10:
                action = 2
        return action


def create_agent(agent_type: str,
                 side: int):
    if side != 1 and side != 2:
        raise Exception("Side can only be 1 or 2")
    if agent_type == 'human':
        return HumanAgent(side=side)
    elif agent_type == 'robot':
        return DQNAgent(side=side)
    elif agent_type == 'god':
        return GodAgent(side=side)
    raise Exception("This agent is not available yet")

