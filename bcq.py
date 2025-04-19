import time
import gymnasium as gym
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

from utils import episode_reward_plot
from offline_data import DDQN
import matplotlib.pyplot as plt


class BCQNetwork(nn.Module):
    """
    Policy and imitation network for discrete BCQ algorithm
    """

    def __init__(self, num_obs, num_actions):
        super(BCQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

        self.imitation = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

    def forward(self, x):
        return self.layers(x), F.log_softmax(self.imitation(x), dim=1)


class BCQ:
    """
    Discrete BCQ algorithm following https://arxiv.org/abs/1910.01708
    """

    def __init__(self, env, gamma=0.99, learning_rate=3e-4, sync_after=5, batch_size=32, threshold=0.3,
                 val_freq=500, val_episodes=20):
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_after = sync_after
        self.learning_rate = learning_rate
        self.val_freq = val_freq
        self.val_episodes = val_episodes

        self.Q_net = BCQNetwork(num_obs=self.obs_dim, num_actions=self.act_dim)
        self.Q_target_net = BCQNetwork(self.obs_dim, self.act_dim)
        self.Q_target_net.load_state_dict(self.Q_net.state_dict())

        self.optim = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate)

        self.threshold = threshold

    def predict(self, state):
        """Predict the best action based on state using batch-constrained modification

        Returns
        -------
        int
            The action to be taken.
        """

        with torch.no_grad():
            q_values, imts = self.Q_net.forward(state)
            # We used log-softmax in the network construction
            imts = imts.exp()

            maximum_action_prob = torch.max(imts,dim=1)[0]
            imtsmask = (imts/maximum_action_prob.unsqueeze(1)) >= self.threshold

            # Use large negative number to mask actions from argmax
            actions = (imtsmask * q_values + (1 - imtsmask * -1e8)).argmax(1)

        return actions

    def train(self, offline_buffer, trainsteps, baseline=None):
        """
        Train the BCQ algorithm
        """

        val_rewards = []

        if self.batch_size > len(offline_buffer):
            raise RuntimeError('Not enough data in buffer!')

        print('Running BCQ on offline data..')
        time.sleep(1)
        for trainstep in tqdm(range(trainsteps)):

            obs, actions, rewards, next_obs, done = offline_buffer.get(batch_size=self.batch_size)


            next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs])
            next_actions = self.predict(next_obs)

            # compute target q-values
            with torch.no_grad():
                next_q_values, _ = self.Q_target_net(next_obs)

                q_values_chosen_actions = next_q_values.gather(1, torch.LongTensor(next_actions).unsqueeze(1)).squeeze(1)
                expected_q_values = torch.Tensor(rewards) + (self.gamma * q_values_chosen_actions * (1-torch.Tensor(done)))

            obs = torch.stack([torch.Tensor(ob) for ob in obs])
            q_values, imt = self.Q_net(obs)
            q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)

            # compute loss of Q-network and imitation network
            q_loss = F.mse_loss(q_values, expected_q_values)
            i_loss = F.nll_loss(imt, torch.LongTensor(actions))

            Q_loss = q_loss + i_loss

            # perform updates
            self.optim.zero_grad()
            Q_loss.backward()
            self.optim.step()

            # synchronize target network
            if trainstep % self.sync_after == 0:
                self.Q_target_net.load_state_dict(self.Q_net.state_dict())

            # validate with current policy
            if trainstep % self.val_freq == 0:
                rewards = self.validate()
                val_rewards.extend(rewards)
                episode_reward_plot(val_rewards, trainstep, window_size=9, step_size=1, line=baseline)

        return val_rewards

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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', '-pre', type=int, default=20000) 
    parser.add_argument('--offline_data', '-data', type=int, default=20000)
    parser.add_argument('--train', '-train', type=int, default=50000)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0003)
    parser.add_argument('--gamma', '-gamma', type=float, default=0.99)
    parser.add_argument('--threshold', '-thres', type=float, default=0.5) 
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
    data = ddqn.generate_data(data_size=_args.offline_data, epsilon=0)
    threshold = _args.threshold

    # Run discrete BCQ algorithm
    for threshold in [0, 0.3, 0.5, 0.7, 1]:
        bcq = BCQ(_env, threshold=threshold, learning_rate=_args.learning_rate, gamma=_args.gamma)
        bcq.train(data, trainsteps=_args.train, baseline=val)
        plt.savefig("bcq"+str(threshold)+".png")
        valbcq = bcq.validate()
        print('Average reward offline policy : {:.1f}'.format(np.mean(valbcq)))
