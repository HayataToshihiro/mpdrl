import os
import tensorflow as tf
import numpy as np
import gym
from mpdrl.envs import MpdrlEnv
from train import PPONet

GAME = 'mpdrl-v0'
MAX_EP_STEP = 1000
MAX_EP = 10
NN_MODEL = os.path.join(os.path.dirname(__file__), 'models/ppo_model.ckpt')
env = gym.make(GAME)

def main():
    sess = tf.Session()
    with tf.device("/cpu:0"):
        brain = PPONet(sess)
        saver = tf.train.Saver()
        print(NN_MODEL)
        saver.restore(sess, NN_MODEL)
    
        for ep in range(MAX_EP):
            s = env.reset().reshape(-1)
            ep_r = 0
            for t in range(MAX_EP_STEP):
                env.render()

                s = np.array([s])
                a = brain.predict_a(s).reshape(-1)

                s_, r, done, info = env.step(a)
                if t ==  MAX_EP_STEP-1:
                    done = True
                ep_r += r
                s = s_.reshape(-1)
                if done:
                    break
            print(ep, ep_r, done,t)


if __name__ == '__main__':
    main()
