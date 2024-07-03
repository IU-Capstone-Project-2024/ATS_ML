from RL_utils.training_env import KnifeEnv
from RL_utils.reward_functions import calc_profit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the training data
training_files = [
    "../data/raw/dumps_aggregated0.csv",
    "../data/raw/dumps_aggregated1.csv",
    "../data/raw/dumps_aggregated2.csv",
    "../data/raw/dumps_aggregated3.csv",
]
test_data_path = "../data/raw/dumps_aggregated4.csv"
training_dfs = [pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), file)) for file in training_files]
df_test = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), test_data_path))
print(len(training_dfs),len(df_test))

# Create the environment
window_size = 20 # take last 5min=20*15s
training_period = 20
environments = [DummyVecEnv([lambda df=df: KnifeEnv(data=df, window_size=window_size)]) for df in training_dfs]

# learn
model = PPO("MlpPolicy", environments[0])
for env in environments:
    model.set_env(env)
    model.learn(total_timesteps=50_000, progress_bar=True)
# model.save("../models/knifebot_all_data")
# model = PPO.load("../models/knifebot_all_data")
# Evaluate the model, set initial parameters
test_env = KnifeEnv(data=df_test, window_size=window_size)
obs, _ = test_env.reset()
done = False

# iteratively step through whole data
while not done:
    # predict next action
    action, _states = model.predict(obs)
    # act according to action, obtain results
    obs, reward, done, _, info = test_env.step(action)

all_trades=info['trades']
all_rewards=info['rewards']

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
prices['unix'] = df_test['unix']
prices['Close'] = df_test['Close']

plt.plot(prices['unix'], prices['Close'], label='Price')
for trade in all_trades:
    plt.axvline(x=trade.entry_timestamp, color='g', linestyle='--', label='Entry')
    if trade.exit_timestamp==None:
        plt.axvline(x=df_test['unix'].iloc[-1], color='r', linestyle='--', label='Forced Exit')
    else:
        plt.axvline(x=trade.exit_timestamp, color='r', linestyle='--', label='Exit')

plt.xlabel('Unix Timestamp')
plt.ylabel('Price')
plt.title('Price Chart with Entry and Exit Timestamps')
plt.legend()
plt.grid(True)
plt.show()

# TODO:
# debug