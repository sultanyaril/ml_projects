""" Handles the initialization of the game through the command line interface.
"""

import tennis_gym
import pygame
from tqdm import tqdm

MAX_STEPS = 2000
GAME_NUMBER = 10000


def main():
    env = tennis_gym.make("Tennis", max_score=1, FPS=0)
    agent1_type = 'robot'
    agent2_type = 'god'
    agent1 = tennis_gym.agents.create_agent(agent1_type, side=1)
    agent2 = tennis_gym.agents.create_agent(agent2_type, side=2)
    won_game = 0
    lost_game = 0
    draw_game = 0
    for _ in tqdm(range(GAME_NUMBER)):
        state, info = env.reset()
        for _ in range(MAX_STEPS):
            # env.render()
            action1 = agent1.act(state)
            state, reward1, done, _, info = env.step(action1)
            action2 = agent2.act(state) + 3
            state, reward2, done, _, info = env.step(action2)
            if done:
                if reward1 == 5 or reward2 == 5:
                    won_game += 1
                elif reward1 == -5 or reward2 == -5:
                    lost_game += 1
                break
        if not done:
            draw_game += 1
    print("Won: {}, Draw: {}, Lost: {}".format(won_game, draw_game, lost_game))
    env.close()


main()
