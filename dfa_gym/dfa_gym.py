import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dfa_samplers import gen_rad

__all__ = ["DFAEnv"]

class DFAEnv(gym.Wrapper):
    def __init__(self, env=None, n_tokens=10, label_f=None, reward_f=None, gen_dfa=None):
        if env is not None: super().__init__(env)
        else:
            self.env = None
            self.action_space = spaces.Discrete(n_tokens)
            self.observation_space = spaces.Discrete(1)
        self.n_tokens = n_tokens
        self.label_f = label_f if label_f is not None else lambda _: np.random.randint(n_tokens)
        self.reward_f = reward_f if reward_f is not None else lambda _, x: x
        self.gen_dfa = gen_dfa if gen_dfa is not None else gen_rad(n_tokens=n_tokens)
        self.dfa = None
    
    def reset(self, *args, **kwargs):
        if self.env is not None:
            obs, info = self.env.reset(*args, **kwargs)
            return (obs, self.dfa), info
        self.dfa = next(self.gen_dfa)
        return self.dfa, {}
    
    def step(self, action):
        if self.env is not None:
            obs, reward, done, truncated, info = self.env.step(action)
            token = self.label_f(obs)
            assert token % self.n_tokens == token
            self.dfa = self.dfa.advance([token]).minimize()
            dfa_reward = self.get_dfa_reward(self.dfa)
            done = done or dfa_reward != 0
            reward = self.reward_f(reward, dfa_reward)
            return (obs, self.dfa), reward, done, truncated, info
        assert action % self.n_tokens == action
        self.dfa = self.dfa.advance([action]).minimize()
        reward = self.get_dfa_reward(self.dfa)
        done = reward != 0
        return self.dfa, reward, done, False, {}

    def close(self):
        if self.env is not None:
            return self.env.close()
        return

    def get_dfa_reward(self, dfa):
        if dfa._label(dfa.start):
            return 1
        if dfa.find_word() is None:
            return -1
        return 0
