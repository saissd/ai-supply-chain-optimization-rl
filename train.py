from stable_baselines3 import DQN
from environment.supply_chain_env import SupplyChainEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Create Environment
env = SupplyChainEnv()
vec_env = DummyVecEnv([lambda: Monitor(env, filename=None)])

# Train DQN Model
model = DQN("MlpPolicy", vec_env, verbose=1, learning_rate=0.001, batch_size=32)
model.learn(total_timesteps=5000)

# Save the trained model
model.save("models/supply_chain_dqn")
