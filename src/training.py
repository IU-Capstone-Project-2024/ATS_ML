from RL_utils.training_env import KnifeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json

n_episodes_to_train = 5_000
script_path = os.path.dirname(os.path.abspath(__file__))
# Load the training data
training_data_paths = [f"../data/raw/dumps_aggregated{train_id}.csv" for train_id in range(0, 6)]
train_data = [pd.read_csv(os.path.join(script_path, path)) for path in training_data_paths]

# # Create the environment
window_size = 40 # 10 minutes
episode_length = 60 # 15 minutes
train_env = KnifeEnv(data=train_data, window_size=window_size, episode_length=episode_length, episode_temp=10)

# # Create one vectorized environment
vec_env = DummyVecEnv([lambda: train_env])

# # Create the model
if os.path.exists(os.path.join(script_path, f"../models/PPO_knife_{window_size}_{episode_length}.zip")):
    print("model loaded")
    model = PPO.load(os.path.join(script_path, f"../models/PPO_knife_{window_size}_{episode_length}.zip"))
    model.set_env(vec_env)
else:
    print("new model created")
    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=4096, ent_coef=0.01)

# Set the tensorboard log directory
models_log_dir = os.path.join(script_path, "../logs/models")
model.tensorboard_log = models_log_dir

# Train the model
model.learn(total_timesteps=n_episodes_to_train * episode_length, progress_bar=True, log_interval=1, tb_log_name="PPO")

# Save the model
model_path = os.path.join(script_path, f"../models/PPO_knife_{window_size}_{episode_length}.zip")
model.save(model_path)

environments_log_dir = os.path.join(script_path, "../../logs/environments")
# Save episode managers stats dicts to json
with open(os.path.join(environments_log_dir, "train_env1_stats.json"), "w") as f:
    json.dump(train_env.episode_manager.stats, f)

# TODO:
# It is hard for RL to understand patterns, need to preprocess data to flags: large movement, large taker volume, half movement of dump