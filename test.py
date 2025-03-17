from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.supply_chain_env import SupplyChainEnv
import numpy as np

# ✅ Create Environment Before Loading Model
env = SupplyChainEnv()
vec_env = DummyVecEnv([lambda: Monitor(env)])

# ✅ Load Model with Environment
model = DQN.load("models/supply_chain_dqn", env=vec_env)


def test_model():
    """Function to test the trained model."""
    obs, _ = vec_env.reset()

    for _ in range(10):  # Run 10 test steps
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)

        assert isinstance(action, np.ndarray), "Action should be a NumPy array"
        assert isinstance(reward, np.ndarray), "Reward should be a NumPy array"
        assert isinstance(done, np.ndarray), "Done should be a NumPy array"

        if done:
            obs, _ = vec_env.reset()

    print("✅ Model successfully tested!")


if __name__ == "__main__":
    test_model()
