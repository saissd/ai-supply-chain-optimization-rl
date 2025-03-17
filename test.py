from stable_baselines3 import DQN
from train import env
import numpy as np

def test_model():
    """Function to test the trained model."""
    model = DQN.load("models/supply_chain_dqn", env=env)
    
    obs, _ = env.reset()  # ✅ Extract only the observation

    for _ in range(20):  # Run 20 test steps
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)  # ✅ Fix unpacking of step output

        assert isinstance(action, np.ndarray), "Action should be a NumPy array"
        assert isinstance(reward, (float, np.ndarray)), "Reward should be a NumPy array or float"
        assert isinstance(done, (bool, np.ndarray)), "Done should be a NumPy array or bool"

        if np.any(done):  # ✅ Fix for multi-env handling
            obs, _ = env.reset()

    print("✅ Model successfully tested!")

if __name__ == "__main__":
    test_model()
