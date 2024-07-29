import gym
from gym import spaces
import numpy as np

class FlightEnv(gym.Env):
    def __init__(self, data):
        super(FlightEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.data.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0
        reward = self._take_action(action)
        done = self.current_step == (len(self.data) - 1)
        return self.data[self.current_step], reward, done, {}

    def _take_action(self, action):
        # Define the reward mechanism based on the action
        reward = 1 if action == 1 else -1
        return reward

    def render(self, mode='human', close=False):
        pass
