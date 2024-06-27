from src.RL_utils.training_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import os

# Load the training data
training_data_path = "../data/BTCUSDT_1m/raw/BTCUSDT_1m_data.csv"
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), training_data_path)
data = pd.read_csv(file_path)

# Create the environment
window_size = 50
training_period = 300
env = TrainEnv(data=data, window_size=window_size, training_period=training_period)

# Create the vectorized environment
vec_env = DummyVecEnv([lambda: env])

# Create the model
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=100000, progress_bar=True)

# Save the model
model.save("../models/RL_trading_model")

# Load the model
model = PPO.load("../models/RL_trading_model")

# Evaluate the model
obs, _ = env.reset()
done = False
reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, r, done, _, _ = env.step(action)
    reward += r

mean_reward = reward / env.total_transactions
print(f"Mean reward: {mean_reward}")

