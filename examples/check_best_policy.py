import numpy as np
import gym
import torch
import os
import time
from mpdrl.envs import MpdrlEnv
from simple_net import PolNet
from machina.pols import GaussianPol

env_name = 'mpdrl-v0'
env = gym.make(env_name)

ob_space = env.observation_space
ac_space = env.action_space

pol_net = PolNet(ob_space, ac_space)

best_path = "garbage/models/pol_max.pkl"
best_pol = GaussianPol(ob_space, ac_space, pol_net)
best_pol.load_state_dict(torch.load(best_path))

done = False
o = env.reset()
for _ in range(300):
    if done:
        time.sleep(0.01)
        o = env.reset()
    ac_real, ac, a_i = best_pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float))
    ac_real = ac_real.reshape(best_pol.ac_space.shape)
    next_o, r, done, e_i = env.step(np.array(ac_real))

    o = next_o
    time.sleep(1/15)
    env.render()