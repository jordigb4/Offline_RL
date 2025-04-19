from collections import deque
import torch
import torch.nn as nn
#from networks import DDQN
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import matplotlib.pyplot as plt


import time
import gymnasium as gym
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR

from utils import episode_reward_plot
from offline_data import DDQN, DQNNetwork


class CQLAgent():

    def __init__(self, env, gamma=0.99, learning_rate=3e-4, sync_after=5, batch_size=32, alpha=1,
                 val_freq=500, val_episodes=20):
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_after = sync_after
        self.learning_rate = learning_rate
        self.val_freq = val_freq
        self.val_episodes = val_episodes
        self.alpha = alpha

        self.Q_net = DQNNetwork(num_obs=self.obs_dim, num_actions=self.act_dim)
        self.Q_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.Q_target_net.load_state_dict(self.Q_net.state_dict())

        self.optim = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate/self.alpha) 
        self.scheduler = MultiplicativeLR(self.optim, lr_lambda=lambda _: 0.99999)


    def predict(self, state):
        with torch.no_grad():
            action_values = self.Q_net(state)
        return torch.argmax(action_values, axis=1)
    

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()


    def validate(self):
        """
        Validate with given policy for 10000 timesteps
        """

        collected_rewards = []
        for _ in range(self.val_episodes):
            obs, _ = self.env.reset()
            timestep_counter = 0
            while True:
                with torch.no_grad():
                    action = self.predict(torch.Tensor(obs).unsqueeze(0))
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                timestep_counter += 1
                if terminated or truncated:
                    collected_rewards.append(timestep_counter)
                    break
        return collected_rewards


    def train(self, offline_buffer, trainsteps, baseline=None):

        val_rewards = []

        if self.batch_size > len(offline_buffer):
            raise RuntimeError('Not enough data in buffer!')

        print('Running CQL on offline data..')
        time.sleep(1)
        for trainstep in tqdm(range(trainsteps)):

            obs, actions, rewards, next_obs, done = offline_buffer.get(self.batch_size)

            next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs])
            next_actions = self.predict(next_obs)
            actions = torch.LongTensor(actions).unsqueeze(1)

            # compute target q-values
            with torch.no_grad():
                next_q_values = self.Q_target_net(next_obs)
                q_values_chosen_actions = next_q_values.gather(1, torch.LongTensor(next_actions).unsqueeze(1)).squeeze(1)
                expected_q_values = torch.Tensor(rewards) + (self.gamma * q_values_chosen_actions * (1 - torch.Tensor(done)))
                
            obs = torch.stack([torch.Tensor(ob) for ob in obs])
            q_values_all= self.Q_net(obs)
            q_values = q_values_all.gather(1, actions).squeeze(1)

            # compute loss of Q-network and imitation network
            q_loss = F.mse_loss(q_values, expected_q_values)
            cql1_loss = self.cql_loss(q_values_all, actions)

            cql2_loss =  (1/(2*self.alpha)) * torch.mean(q_loss)
            CQL_loss = cql1_loss +  cql2_loss

            # perform updates
            self.optim.zero_grad()
            CQL_loss.backward()
            self.optim.step()

            self.scheduler.step()

            # synchronize target network
            if trainstep % self.sync_after == 0:
                self.Q_target_net.load_state_dict(self.Q_net.state_dict())

            # validate with current policy
            if trainstep % self.val_freq == 0:
                rewards = self.validate()
                val_rewards.extend(rewards)
                episode_reward_plot(val_rewards, trainstep, window_size=7, step_size=1, line=baseline)

        return val_rewards


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', '-pre', type=int, default=20000)
    parser.add_argument('--offline_data', '-data', type=int, default=20000)
    parser.add_argument('--train', '-train', type=int, default=25000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--alpha', '-alpha', type=float, default=8) 
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    _args = parse()
    _env = gym.make('CartPole-v1')

    # Pre-train policy for data generation
    ddqn = DDQN(_env)
    res = ddqn.train(timesteps=_args.pre_train)

    # Let's check how the trained (online) policy performs
    val = ddqn.validate()
    print('Average reward using policy for data generation: {:.1f}'.format(val))

    # Generate offline samples
    data = ddqn.generate_data(data_size=_args.offline_data)

    # Run discrete CQL algorithm
    for alpha in [0.5, 1, 2, 4, 8]:
        cql = CQLAgent(_env, learning_rate=_args.learning_rate, gamma=_args.gamma, alpha=alpha)
        cql.train(data, trainsteps=_args.train, baseline=val)
        cqlval = cql.validate()
        plt.savefig("cql"+str(alpha)+".png")
        print('Average reward offline policy : {:.1f}'.format(np.mean(cqlval)))
