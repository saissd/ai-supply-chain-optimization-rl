from stable_baselines3 import DQN
from environment.supply_chain_env import SupplyChainEnv
import numpy as np

# ✅ Load Model with Environment
model = DQN.load("models/supply_chain_dqn", env=vec_env)
model.observation_space = vec_env.observation_space
model.action_space = vec_env.action_space


def test_model():
    # Create environment
    env = SupplyChainEnv()
    obs, _ = env.reset()
    # Run model for 20 test steps
    for _ in range(20):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

    print("✅ Model successfully tested!")


if __name__ == "__main__":
    test_model()
