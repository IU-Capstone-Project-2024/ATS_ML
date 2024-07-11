from RL_utils.training_env import KnifeEnv
from stable_baselines3 import PPO
import pandas as pd
import os
import matplotlib.pyplot as plt

window_size = 80 # 20 minutes
episode_length = 80 # 20 minutes
script_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_path, f"../models/PPO_knife_cumul_80_80.zip")

testing_data_path = "../data/raw/dumps_aggregated7.csv"
test_data = pd.read_csv(os.path.join(script_path, testing_data_path))

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
