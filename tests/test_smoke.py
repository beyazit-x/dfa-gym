from dfa_gym import DFAEnv

if __name__ == "__main__":
    dfa_env = DFAEnv()
    obs, info = dfa_env.reset()
    for _ in range(1000):
        action = dfa_env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = dfa_env.step(action)
        if done:
            break
    dfa_env.close()
