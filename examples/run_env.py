import numpy as np
import gym
from mpdrl.envs import MpdrlEnv

env = gym.make('mpdrl-v0')

for i_episode in range(100):
    observation = env.reset()
    done = False
    ep_r = 0
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(observation)
        ep_r += reward
        if done:
            print("Episode %d  finished after %d timesteps, reward: %f "% (i_episode+1, t+1,ep_r))
            break
