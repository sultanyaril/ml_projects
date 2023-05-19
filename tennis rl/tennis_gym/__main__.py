""" Handles the initialization of the game through the command line interface.
"""

import tennis_gym
import pygame


def main():
    args = tennis_gym.get_args()
    env = tennis_gym.make("Tennis")
    agent1_type = args.player1
    agent2_type = args.player2
    agent1 = tennis_gym.agents.create_agent(agent1_type, side=1)
    agent2 = tennis_gym.agents.create_agent(agent2_type, side=2)
    state, info = env.reset()
    done = False
    quit_flg = False
    while not done and not quit_flg:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flg = True
            if agent1_type == 'human':
                action1 = agent1.act(event)
            if agent2_type == 'human':
                action2 = agent2.act(event) + 3

        if agent1_type != 'human':
            action1 = agent1.act(state)
        if agent2_type != 'human':
            action2 = agent2.act(state) + 3

        state, reward1, done, _, info = env.step(action1)
        state, reward2, done, _, info = env.step(action2)

    env.close()

main()
