import sys
import string
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import RADSampler

__all__ = ["DFAEnv"]

class DFAEnv(gym.Env):
    def __init__(self, sampler=None):
        super().__init__()
        self.sampler = sampler if sampler is not None else RADSampler()
        self.size_bound = self.sampler.get_size_bound()
        print(self.size_bound)
        self.action_space = spaces.Discrete(self.sampler.n_tokens)
        self.observation_space = spaces.Box(low=0, high=9, shape=(self.size_bound,), dtype=np.int64)
        self.dfa = None
    
    def reset(self, seed=None):
        self.dfa = self.sampler.sample()
        return self._get_dfa_obs(), {}
    
    def step(self, action):
        self.dfa = self.dfa.advance([action]).minimize()
        reward = 0
        if self.dfa._label(self.dfa.start):
            reward = 1
        elif self.dfa.find_word() is None:
            reward = -1
        done = reward != 0
        return self._get_dfa_obs(), reward, done, False, {}

    def _get_dfa_obs(self):
        dfa_obs = np.array([int(i) for i in str(self.dfa.to_int())])
        return np.pad(dfa_obs, (self.size_bound - dfa_obs.shape[0], 0), constant_values=0)

