# This file defines the main function: 
# to run the trained model on the provided data and send the results to another endpoint
# The model must send the results to the endpoint in predefined intervals. For example, every 15 seconds.
# The model sends the action to the endpoint and receives confirmation or rejection if endpoint accepts the action.

from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import json

cfg = json.load(open("configs/main.json"))


def wrap_observations(observations, current_gain, holding_flag):
    # Wraps the observations in the format expected by the model 

    if observations.shape != (cfg["observation_window"], len(cfg["observed_features"])):
        raise ValueError(f"Expected observations to have shape {(cfg['observation_window'], len(cfg['observed_features']))} but got {observations.shape}")

    obs = observations[cfg["observed_features"]].values.flatten()
    obs = np.append(obs, [current_gain, holding_flag])

    return obs

def main():
    # Load the model
    model = PPO.load(cfg["model_path"])

    # Define the next prediction time
    # Round the current time to the nearest 15 seconds
    next_prediction_time = np.ceil(time.time() / cfg["prediction_frequency"]) * cfg["prediction_frequency"]
    
    while True:
        # Get the current time
        current_time = time.time()

        # If the current time is greater than the next prediction time, make a prediction
        if current_time >= next_prediction_time:
            
            observations = pd.DataFrame(columns=cfg["observed_features"], data=np.random.randn(cfg["observation_window"], len(cfg["observed_features"])))
            current_gain = 0
            holding_flag = 0

            # Wrap the observations
            obs = wrap_observations(observations, current_gain, holding_flag)

            # Make a prediction
            action, _ = model.predict(obs, deterministic=True)

            # Send the action to the endpoint
            # For now, we will just print the action
            if cfg["receiver_endpoint"]:
                pass
            else:
                print(f"Action: {action}")

            # Update the next prediction time
            next_prediction_time = np.ceil(time.time() / cfg["prediction_frequency"]) * cfg["prediction_frequency"]

        # Sleep for 1 second
        time.sleep(0.1)
            
    

if __name__ == "__main__":
    main()