import tennis_gym
import numpy as np
import random
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import deque

EPISODES = 10000
WARM_UP = 50
ANNEAL_EPISODES = EPISODES - WARM_UP
SYNC_TARGET_EPISODES = 10
MAX_STEPS = 2000
BATCH_SIZE = 32
MEMORY_SIZE = 10000
GAMMA = 0.99
EPSILON = 1
MIN_EPSILON = 0.02
LEARNING_RATE = 0.0001
INITIAL_MODEL_FILE = 'tennis_gym/models/32batch_10000memory_30000episodes'
start_episode = 0
if INITIAL_MODEL_FILE != '':
    start = INITIAL_MODEL_FILE.rfind('_') + 1
    end = INITIAL_MODEL_FILE.find('episodes')
    start_episode = int(INITIAL_MODEL_FILE[start:end])
print('FIRST EPISODE IS ', start_episode)



SCREEN_SIZE = (500, 250)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        self.relu_type = type(nn.ReLU())

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, epochs=1):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.net(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def predict(self, X):
        with torch.no_grad():
            result = self.net(X)
        return result

    def get_weights(self):
        weights = []
        for i in self.net:
            if type(i) == self.relu_type:
                weights.append(0)
                continue
            weights.append(i.weight)
        return weights

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if type(self.net[i]) == self.relu_type:
                continue
            self.net[i].weight = weight

    def save(self, path):
        torch.save(self.net.state_dict(), path)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self._build_model()

    def _build_model(self):
        return Model(self.state_size, self.action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.Tensor(state.reshape(1, len(state)))
        act_values = self.model.predict(state).detach().numpy()
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) == MEMORY_SIZE - 1:
            print("I'm full")
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size, current_episode):
        if batch_size < len(self.memory):
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        states = torch.Tensor(np.array([sample[0] for sample in minibatch]))
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = torch.Tensor(np.array([sample[3] for sample in minibatch]))
        dones = np.array([sample[4] for sample in minibatch])

        targets = self.target_model.predict(states).detach().numpy()
        Q_futures = np.amax(self.target_model.predict(next_states).detach().numpy(), axis=1)
        target_actions = actions % 3
        target_rewards = rewards + (np.logical_not(dones)) * (GAMMA * Q_futures)

        for i in range(len(minibatch)):
            targets[i][target_actions[i]] = target_rewards[i]

        targets = torch.Tensor(targets)
        loss = self.model.fit(states, targets, epochs=1)
        if current_episode > WARM_UP:
            self.epsilon = max(MIN_EPSILON,
                               EPSILON - (e - WARM_UP + 1) * (EPSILON - MIN_EPSILON) / ANNEAL_EPISODES)
        return loss

    def target_train(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)


def swap_paddles(obs):
    """ Reverses field so model train only one paddle"""
    obs[0] = SCREEN_SIZE[0] - obs[0]
    obs[2] *= -1
    obs[5], obs[7] = obs[7], obs[5]
    return obs


def init_agent(agent):
    if INITIAL_MODEL_FILE != "":
        agent.model.net.load_state_dict(torch.load(INITIAL_MODEL_FILE))
    return agent


env = tennis_gym.make("Tennis", screen_size=SCREEN_SIZE, max_score=1, FPS=0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent = init_agent(agent)

mse = None
total_mse = 0
i = 1
for e in (pbar := tqdm(range(EPISODES))):
    mse = total_mse / i
    pbar.set_description("Epsilon: {:.3f} MSE: {:.3f}".format(agent.epsilon, mse))
    state1, info = env.reset()
    state2 = swap_paddles(state1)
    done = False
    total_reward = 0
    total_mse = 0
    for i in range(MAX_STEPS):
        env.render()
        action1 = agent.act(state1)
        action2 = agent.act(state2) + 3
        next_state1, reward1, done, _, info = env.step(action1)
        next_state2, reward2, done, _, info = env.step(action2)
        tmp_reward1 = (reward1 == -5 or reward2 == -5) * -20000 + (reward1 == 1 or reward2 == 1) * 100000
        tmp_reward2 = (reward1 == 5 or reward2 == 5) * 20000 + (reward1 == -1 or reward2 == -1) * -100000
        reward1 = tmp_reward1
        reward2 = tmp_reward2
        reward = reward1 + reward2
        next_state2 = swap_paddles(next_state2)
        agent.remember(state1, action1, reward1, next_state1, done)
        agent.remember(state2, action2, -reward2, next_state2, done)
        total_reward += reward

        state1 = next_state1
        state2 = next_state2
        total_mse += agent.replay(BATCH_SIZE, e)

        if done:
            break
    if (e + 1) % 10:
        agent.target_train()
    if (e + 1) % 500 == 0:
        agent.model.save('tennis_gym/models/{}batch_{}memory_{}episodes'.format(BATCH_SIZE, MEMORY_SIZE, start_episode + e + 1))
    # print the scores after every episode
