import flappy_bird_gym

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

import warnings
warnings.simplefilter("ignore")
import sys

def build_model(obs, actions):
    model = Sequential()
    
    model.add(Dense(64, activation='relu', input_shape=(1, obs)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    # model.summary()
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.2, value_min=.0001, value_test=.0, nb_steps=6000000)
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                enable_dueling_network=True, dueling_type='avg',
                nb_actions=actions, nb_steps_warmup=500)
    return dqn



if __name__ == '__main__':
    make_gif=False
    background='night'

    if len(sys.argv) == 2:
        make_gif = sys.argv[1]
    elif len(sys.argv) == 3:
        background = sys.argv[2]

    env = flappy_bird_gym.make("FlappyBird-v0", make_gif=make_gif, background=background)
    obs = env.observation_space.shape[0]
    actions = env.action_space.n

    model = build_model(obs, actions)
    dqn = build_agent(model, actions)

    dqn.compile(Adam(learning_rate=0.00025))

    dqn.load_weights("weights/flappy_bird_solution_simple.h5")

    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()