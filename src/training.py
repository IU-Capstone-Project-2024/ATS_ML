from RL_utils.training_env import SparseTrainEnv
from RL_utils.reward_functions import calc_profit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import json

n_episodes_to_train = 3000
model_name = "FinalPPO"

script_path = os.path.dirname(os.path.abspath(__file__))

# Load the training data
training_data_paths1 = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(0, 6)]
train_data1 = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths1]

training_data_paths2 = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(6, 14)]
train_data2 = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths2]

training_data_paths3 = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(14, 20)]
train_data3 = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths3]

training_data_paths4 = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(20, 27)]
train_data4 = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths4]

# Load the test data
test_data_path = "../data/raw/dumps_aggregated10.csv"
test_data = pd.read_csv(os.path.join(script_path, test_data_path))

# Create the environment
window_size = 20 # 5 minutes
episode_length = 80 # 20 minutes
train_env1 = SparseTrainEnv(data=train_data1, window_size=window_size, episode_length=episode_length, episod_temp=7)
train_env2 = SparseTrainEnv(data=train_data2, window_size=window_size, episode_length=episode_length, episod_temp=7)
train_env3 = SparseTrainEnv(data=train_data3, window_size=window_size, episode_length=episode_length, episod_temp=7)
train_env4 = SparseTrainEnv(data=train_data4, window_size=window_size, episode_length=episode_length, episod_temp=7)


# Create one vectorized environment
vec_env = DummyVecEnv([lambda: train_env1, lambda: train_env2, lambda: train_env3, lambda: train_env4])

# Create the model
if os.path.exists(os.path.join(script_path, f"../models/{model_name}_{window_size}_{episode_length}_trained.zip")):
    model = PPO.load(os.path.join(script_path, f"../models/{model_name}_{window_size}_{episode_length}_trained.zip"))
    model.set_env(vec_env)
# else:
#     model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=4096)

# # Set the tensorboard log directory
# models_log_dir = os.path.join(script_path, "../logs/models")
# model.tensorboard_log = models_log_dir

# # Train the model
# model.learn(total_timesteps=n_episodes_to_train * episode_length, progress_bar=True, log_interval=1, tb_log_name=model_name)

# Save the model
# model_path = os.path.join(script_path, f"../models/{model_name}_{window_size}_{episode_length}.zip")
# model.save(model_path)

# environments_log_dir = os.path.join(script_path, "../logs/environments")
# # Save episode managers stats dicts to json
# with open(os.path.join(environments_log_dir, "train_env1_stats.json"), "w") as f:
#     json.dump(train_env1.episode_manager.stats, f)
# with open(os.path.join(environments_log_dir, "train_env2_stats.json"), "w") as f:
#     json.dump(train_env2.episode_manager.stats, f)

# Load the model
# model = PPO.load(model_path)

# Evaluate the model
test_env = SparseTrainEnv(data=[test_data], window_size=window_size, test_mode=True, transaction_cost_pct=0)
obs, _ = test_env.reset()
done = False
reward = 0
reward_history = []
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info, _ = test_env.step(action)
    reward += rewards
    reward_history.append(rewards)

print(f"Total reward: {reward}")
print(f"Mean reward: {reward /( len(test_data) - 2 - window_size)}")

test_history = pd.DataFrame(test_env.past_actions, columns=["timestamp", "action", "price", "balance", "tokens_held"])
net_worth_history = test_history["balance"] + test_history["tokens_held"] * test_history["price"]

price_scaled = test_history["price"] / test_history["price"].iloc[0]
net_worth_scaled = net_worth_history / net_worth_history.iloc[0]

# Plot the test history with buy and sell signals and rewards history on the second plot
plt.figure(figsize=(30, 30))
plt.subplot(2, 1, 1)
plt.plot(test_history["timestamp"], price_scaled, label="Price")
plt.plot(test_history["timestamp"], net_worth_scaled, label="Net Worth")
buy_timestamps = test_history[test_history["action"] == "Buy"]["timestamp"]
sell_timestamps = test_history[test_history["action"] == "Sell"]["timestamp"]
plt.vlines(buy_timestamps, ymin=min(price_scaled.min(), net_worth_scaled.min()), ymax=max(price_scaled.max(), net_worth_scaled.max()), colors='g', linestyles='dashed', label='Buy')
plt.vlines(sell_timestamps, ymin=min(price_scaled.min(), net_worth_scaled.min()), ymax=max(price_scaled.max(), net_worth_scaled.max()), colors='r', linestyles='dashed', label='Sell')
plt.subplot(2, 1, 2)
plt.plot(test_history["timestamp"], reward_history, label="Reward")
plt.vlines(buy_timestamps, ymin=min(reward_history), ymax=max(reward_history), colors='g', linestyles='dashed', label='Buy')
plt.vlines(sell_timestamps, ymin=min(reward_history), ymax=max(reward_history), colors='r', linestyles='dashed', label='Sell')
plt.legend()
plt.show()


