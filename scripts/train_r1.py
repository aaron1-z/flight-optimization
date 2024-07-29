from stable_baselines3 import PPO
from envs.flight_env import FlightEnv
import numpy as np

def main():
    # Generate synthetic data
    data = np.random.rand(1000, 10)
    
    # Train RL model
    env = FlightEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("rl_model")

if __name__ == "__main__":
    main()
