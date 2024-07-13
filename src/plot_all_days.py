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


model_name = "FinalPPO"

script_path = os.path.dirname(os.path.abspath(__file__))

# Load the test data
test_data_dir = "../data/raw/"

# Create the environment
window_size = 20 # 5 minutes
episode_length = 80 # 20 minutes

# Load the model
if os.path.exists(os.path.join(script_path, f"../models/{model_name}_{window_size}_{episode_length}_trained.zip")):
    model = PPO.load(os.path.join(script_path, f"../models/{model_name}_{window_size}_{episode_length}_trained.zip"))
else:
    exit(1)

# Evaluate the model
for i in range(28):
    test_df = pd.read_csv(os.path.join(script_path, test_data_dir, f"dumps_aggregated{i}.csv"))
    test_env = SparseTrainEnv(data=[test_df], window_size=window_size, test_mode=True, transaction_cost_pct=0)
    obs, _ = test_env.reset()
    done = False
    reward = 0
    reward_history = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info, _ = test_env.step(action)
        reward += rewards
        reward_history.append(rewards)


    test_history = pd.DataFrame(test_env.past_actions, columns=["timestamp", "action", "price", "balance", "tokens_held"])
    net_worth_history = test_history["balance"] + test_history["tokens_held"] * test_history["price"]

    price_scaled = test_history["price"] / test_history["price"].iloc[0]
    net_worth_scaled = net_worth_history / net_worth_history.iloc[0]

    # Plot the test history with buy and sell signals and rewards history on the second plot
    plt.figure(figsize=(30, 15))
    plt.plot(test_history["timestamp"], price_scaled, label="Price")
    plt.plot(test_history["timestamp"], net_worth_scaled, label="Net Worth")
    buy_timestamps = test_history[test_history["action"] == "Buy"]["timestamp"]
    sell_timestamps = test_history[test_history["action"] == "Sell"]["timestamp"]
    plt.vlines(buy_timestamps, ymin=min(price_scaled.min(), net_worth_scaled.min()), ymax=max(price_scaled.max(), net_worth_scaled.max()), colors='g', linestyles='dashed', label='Buy')
    plt.vlines(sell_timestamps, ymin=min(price_scaled.min(), net_worth_scaled.min()), ymax=max(price_scaled.max(), net_worth_scaled.max()), colors='r', linestyles='dashed', label='Sell')
    plt.ylabel("Day profit")
    plt.xlabel("Time")

    plt.legend()
    plt.savefig(os.path.join(script_path, f"../reports/{model_name}_{window_size}_{episode_length}_day{i}.png"))


