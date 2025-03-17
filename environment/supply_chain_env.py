
import gymnasium as gym  # ✅ Ensure gymnasium is imported properly
from gymnasium import spaces
import numpy as np
# ✅ Custom Supply Chain Environment (Fixed Inheritance)
class SupplyChainEnv(gym.Env):  # ✅ Explicitly inherit from gym.Env
    def __init__(self):
        super(SupplyChainEnv, self).__init__()

        # ✅ Define Action Space: {0: Reduce Stock, 1: Maintain, 2: Increase Stock}
        self.action_space = spaces.Discrete(3)

        # ✅ Define Observation Space: Inventory level (bounded between 0 and 100)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        # ✅ Initial inventory level
        self.state = np.array([50], dtype=np.float32)

        # ✅ Maximum simulation steps
        self.max_steps = 50
        self.current_step = 0

    def step(self, action):
        """
        Take an action in the environment and update inventory state.
        """
        self.current_step += 1

        # Action Mapping:
        if action == 0:  # Reduce Stock
            self.state[0] -= 10
        elif action == 2:  # Increase Stock
            self.state[0] += 10

        # ✅ Demand Simulation: Randomized demand between 5 and 15 units
        demand = np.random.randint(5, 15)
        self.state[0] -= demand  # Deduct demand from stock
        self.state[0] = np.clip(self.state[0], 0, 100)  # Ensure inventory stays within bounds

        # ✅ Reward Function: Encourage inventory level close to 50
        reward = -abs(50 - self.state[0])  # Best reward when inventory ≈ 50

        # ✅ Termination Condition
        done = self.current_step >= self.max_steps

        return self.state, reward, done, False, {}  # ✅ Follow Gymnasium's API

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        self.state = np.array([50], dtype=np.float32)
        self.current_step = 0
        return self.state, {}  # ✅ Must return (obs, info) in Gymnasium

    def render(self):
        """
        Render the environment (prints inventory level).
        """
        print(f"Step: {self.current_step}, Inventory Level: {self.state[0]}")
