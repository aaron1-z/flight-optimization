from stable_baselines3 import PPO
from envs.flight_env import FlightEnv

def train_rl_model(data, timesteps=10000):
    env = FlightEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
