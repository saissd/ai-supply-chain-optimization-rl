import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from environment.supply_chain_env import SupplyChainEnv  # Import custom environment

# ✅ Create Environment & Wrap it Properly (Fixed Wrapping Issue)
env = SupplyChainEnv()
vec_env = DummyVecEnv([lambda: Monitor(env, filename=None)])  # ✅ Prevent logging errors

# ✅ Train DQN Model
model = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=0.001, batch_size=32)
model.learn(total_timesteps=5000)

# ✅ Save the trained model
model.save("supply_chain_dqn")

print("✅ Training complete! Model saved.")
