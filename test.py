from stable_baselines3 import DQN
from environment.supply_chain_env import SupplyChainEnv
import gymnasium
# Load trained model
model = DQN.load("models/supply_chain_dqn")

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
