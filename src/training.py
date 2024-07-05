from RL_utils.training_env import TrainEnv, SparseTrainEnv, KnifeEnv
from RL_utils.reward_functions import calc_profit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import json

n_episodes_to_train = 10_000

script_path = os.path.dirname(os.path.abspath(__file__))

# Load the training data
training_data_paths = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(0, 3)]
train_data = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths]

# Load the test data
test_data_path = "../data/raw/dumps_aggregated4.csv"
test_data = pd.read_csv(os.path.join(script_path, test_data_path))

# # Create the environment
window_size = 20 # 5 minutes
episode_length = 40 # 10 minutes
train_env = KnifeEnv(data=train_data, window_size=window_size, episode_length=episode_length)

# # Create one vectorized environment
vec_env = DummyVecEnv([lambda: train_env])

# # Create the model
if os.path.exists(os.path.join(script_path, f"../models/PPO_knife_{window_size}_{episode_length}_trained.zip")):
    model = PPO.load(os.path.join(script_path, f"../models/PPO_knife_{window_size}_{episode_length}_trained.zip"))
    model.set_env(vec_env)
else:
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=4096, ent_coef=0.01)

# # Set the tensorboard log directory
models_log_dir = os.path.join(script_path, "../logs/models")
model.tensorboard_log = models_log_dir

# Train the model
model.learn(total_timesteps=n_episodes_to_train * episode_length, progress_bar=True, log_interval=1, tb_log_name="PPO")

# # Save the model
model_path = os.path.join(script_path, f"../models/PPO_{window_size}_{episode_length}.zip")
model.save(model_path)

environments_log_dir = os.path.join(script_path, "../../logs/environments")
# Save episode managers stats dicts to json
with open(os.path.join(environments_log_dir, "train_env1_stats.json"), "w") as f:
    json.dump(train_env.episode_manager.stats, f)

# Load the model
model = PPO.load(model_path)

# Evaluate the model
test_env = KnifeEnv(data=[test_data], window_size=window_size, test_mode=True)
obs, _ = test_env.reset()
done = False
reward = 0

all_trades=[]
all_rewards=[]
# iteratively step through whole data
while not done:
    # predict next action
    action, _states = model.predict(obs)
    # act according to action, obtain results
    obs, reward, done, _, info = test_env.step(action)
    if 'trades' in info.keys():
        all_trades+=info['trades']
        all_rewards+=info['rewards']

# Visualisations
print('Amount of trades:', len(all_trades))
total_profit = 0
for trade in all_trades:
    total_profit += trade.profit
    print(trade.entry_price, trade.exit_price, trade.profit)
print('Total profit:', total_profit)

# Plot rewards
plt.figure(figsize=(14, 7))
plt.plot(all_rewards, marker='o', linestyle='-', color='b', label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.legend()
plt.grid(True)
plt.show()

# Plot price chart with entry and exit timestamps
plt.figure(figsize=(14, 7))
prices = pd.DataFrame()
prices['unix'] = test_data['unix']
prices['Close'] = test_data['Close']

plt.plot(prices['unix'], prices['Close'], label='Price')
for trade in all_trades:
    plt.axvline(x=trade.entry_timestamp, color='g', linestyle='--', label='Entry')
    if trade.exit_timestamp==None:
        plt.axvline(x=test_data['unix'].iloc[-1], color='r', linestyle='--', label='Forced Exit')
    else:
        plt.axvline(x=trade.exit_timestamp, color='r', linestyle='--', label='Exit')

plt.xlabel('Unix Timestamp')
plt.ylabel('Price')
plt.title('Price Chart with Entry and Exit Timestamps')
plt.legend()
plt.grid(True)
plt.show()

# TODO:
# It is hard for RL to understand patterns, need to preprocess data to flags: large movement, large taker volume, half movement of dump
