from RL_utils.training_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the training data
training_data_path = "../data/BTCUSDT_1m/raw/BTCUSDT_1m_data.csv"
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), training_data_path)
data = pd.read_csv(file_path)

train_data = data[:int(0.8 * len(data))]
print(train_data.shape)
test_data = data[int(0.8 * len(data)):]
print(test_data.shape)

# Create the environment
window_size = 50
training_period = 300
train_env = TrainEnv(data=train_data, window_size=window_size, training_period=training_period)

# Create the vectorized environment
vec_env = DummyVecEnv([lambda: train_env])

# Create the model
model = PPO("MlpPolicy", vec_env, verbose=1, ent_coef=0.001)

# Train the model
model.learn(total_timesteps=100000, progress_bar=True)

# Save the model
# model.save("../models/RL_trading_model")

# # Load the model
# model = PPO.load("../models/RL_trading_model")

# Evaluate the model
test_env = TrainEnv(data=test_data, window_size=window_size, test_mode=True)
obs = test_env.reset()
done = False
reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    reward += rewards

print(f"Total reward: {reward}")
print(f"Mean reward: {reward /( len(test_data) - 2 - window_size)}")

test_history = pd.DataFrame(test_env.past_actions, columns=["timestamp", "action", "price", "balance", "tokens_held"])
net_worth_history = test_history["balance"] + test_history["tokens_held"] * test_history["price"]

price_scaled = test_history["price"] / test_history["price"].iloc[0]
net_worth_scaled = net_worth_history / net_worth_history.iloc[0]

plt.figure(figsize=(30, 15))
plt.plot(test_history["timestamp"], price_scaled, label="Price")
plt.plot(test_history["timestamp"], net_worth_scaled, label="Net worth")
plt.legend()
plt.show()
