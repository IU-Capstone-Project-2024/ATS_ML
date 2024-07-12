from RL_utils.training_env import KnifeEnv
from stable_baselines3 import PPO
import pandas as pd
import os

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
print(obs.shape)
action, _states = model.predict(obs)
print(action)