import argparse
import numpy as np
import gym
import torch
import os
import time
from mpdrl.envs import MpdrlEnv
from simple_net import PolNet, PolNetLSTM
from machina.pols import GaussianPol

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='garbage/models/pol_max.pkl',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='mpdrl-v0', help='Name of environment.')
parser.add_argument('--max_epis', type=int,
                    default=3, help='Number of episodes to run.')
parser.add_argument('--max_steps', type=int,
                    default=500, help='Number of steps to run.')
parser.add_argument('--rnn', action='store_true',
                    default=False, help='If True, network is reccurent.')

args = parser.parse_args()

env_name = args.env_name
env = gym.make(env_name)

observation_space = env.observation_space
action_space = env.action_space

if args.rnn:
    pol_net = PolNetLSTM(observation_space, action_space, h_size=256, cell_size=256)
else:
    pol_net = PolNet(observation_space, action_space)

best_path = args.model_path
best_pol = GaussianPol(observation_space, action_space, pol_net, args.rnn)
best_pol.load_state_dict(torch.load(best_path))

done = False
o = env.reset()
for _ in range(args.max_epis):
    for step in range(args.max_steps):
        env.render()
        ac_real, ac, a_i = best_pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float))
        ac_real = ac_real.reshape(best_pol.ac_space.shape)
        next_o, r, done, e_i = env.step(np.array(ac_real))
        if done:
            break
        o = next_o