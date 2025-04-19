import random
import numpy as np
import time
from tqdm import tqdm
from collections import deque

import torch
from torch import nn, optim
import torch.nn.functional as F


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""

        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def put(self, obs, action, reward, next_obs, terminated):
        """Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        """

        self.buffer.append((obs, action, reward, next_obs, terminated))

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""

        return zip(*random.sample(self.buffer, batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""

        return len(self.buffer)


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DDQN:

    def __init__(self, env, gamma=0.99, learning_rate=3e-4, replay_size=10000, sync_after=5, batch_size=32):

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_after = sync_after
        self.learning_rate = learning_rate

        self.replay_buffer = ReplayBuffer(replay_size)

        self.dqn_net = DQNNetwork(num_obs=self.obs_dim, num_actions=self.act_dim)
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=self.learning_rate)

    def predict(self, state, epsilon=1.0):
        """Predict action based on state using Gibbs/Boltzmann sampling
        
        Parameters
        ----------
        state : array-like
            The current state
        temperature : float
            Temperature parameter controls exploration (higher = more exploration)
        
        Returns
        -------
        int
            The selected action
        """
        temperature = 1
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.dqn_net.forward(state)

            q_max = torch.max(q_values)
            q_values = q_values - q_max
            
            # Apply softmax with temperature
            scaled_q = q_values/temperature
            exp_q = torch.exp(scaled_q)
            probabilities = exp_q / torch.sum(exp_q)
            
            # Convert to numpy for sampling
            probs = probabilities.numpy().flatten()

            # Ensure probabilities sum to 1 and are non-negative
            probs = np.clip(probs, 0, 1)
            probs = probs / np.sum(probs)
            
            # Sample action based on probabilities
            action = np.random.choice(self.act_dim, p=probs)
            
        return action

    # def predict(self, state, epsilon=0.0):
    #     """Predict the best action based on state. With probability epsilon take random action

    #     Returns
    #     -------
    #     int
    #         The action to be taken.
    #     """

    #     if random.random() > epsilon:
    #         with torch.no_grad():
    #             state = torch.FloatTensor(state).unsqueeze(0)
    #             q_value = self.dqn_net.forward(state)
    #             action = q_value.argmax().item()
    #     else:
    #         action = random.randrange(self.act_dim)
    #     return action

    def train(self, timesteps, epsilon=0.1):

        obs, _ = self.env.reset()
        step_counter = 0
        collected_rewards = []

        print('Training policy for data generation...')
        time.sleep(1)
        for timestep in tqdm(range(1, timesteps+1)):

            action = self.predict(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            step_counter += 1
            self.replay_buffer.put(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs

            if terminated or truncated:
                collected_rewards.append(step_counter)
                step_counter = 0
                obs, _ = self.env.reset()

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_msbe_loss()

                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

        return collected_rewards

    def compute_msbe_loss(self):

        obs, actions, rewards, next_obs, done = self.replay_buffer.get(self.batch_size)

        obs = torch.stack([torch.Tensor(ob) for ob in obs])
        next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs])

        q_values = self.dqn_net(obs)
        next_q_values = self.dqn_target_net(next_obs)

        q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]

        expected_q_values = torch.Tensor(rewards) + self.gamma * (1.0 - torch.Tensor(done)) * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)
        return loss

    def validate(self):

        obs, _ = self.env.reset()

        timestep_counter = 0
        collected_rewards = []

        for _ in range(10000):

            with torch.no_grad():
                #action = self.predict(torch.Tensor(obs))
                obs = torch.FloatTensor(torch.Tensor(obs)).unsqueeze(0)
                q_value = self.dqn_net.forward(obs)
                action = q_value.argmax().item()

            obs, reward, terminated, truncated, _ = self.env.step(action)
            timestep_counter += 1

            if terminated or truncated:
                collected_rewards.append(timestep_counter)
                timestep_counter = 0
                obs, _ = self.env.reset()

        return np.average(collected_rewards)


    def generate_data(self, data_size, epsilon=0.1):

        offline_buffer = ReplayBuffer(data_size)

        obs, _ = self.env.reset()

        print('Generating offline data...')
        time.sleep(1)
        for _ in tqdm(range(data_size)):

            action = self.predict(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            offline_buffer.put(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs

            if terminated or truncated:
                obs, _ = self.env.reset()

        return offline_buffer
