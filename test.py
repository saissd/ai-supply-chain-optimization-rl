from stable_baselines3 import DQN
from environment.supply_chain_env import SupplyChainEnv
import numpy as np


def test_model():
    """Function to test the trained model."""
    env = SupplyChainEnv()
    model = DQN.load("models/supply_chain_dqn", env=env)
    obs = env.reset()

    for _ in range(20):  # Run 20 test steps
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        assert isinstance(action, np.ndarray), "Action should be a NumPy array"
        assert isinstance(reward, np.ndarray), "Reward should be a NumPy array"
        assert isinstance(done, np.ndarray), "Done should be a NumPy array"

        if done.any():  # Fix for multi-env handling
            obs = env.reset()

    print("✅ Model successfully tested!")


if __name__ == "__main__":
    test_model()
