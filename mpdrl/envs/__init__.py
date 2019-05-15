import gym
from gym.envs.registration import registry

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

from mpdrl.envs.mpdrl import MpdrlEnv

register(
  id='mpdrl-v0',
  entry_point='mpdrl.envs.mpdrl:MpdrlEnv',
  max_episode_steps=500,
)
