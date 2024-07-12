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
from src.easy_data_collection import get_last_data
from src.RL_utils.Order import TestOrder
from src.RL_utils.reward_functions import calc_profit
import zmq

cfg = json.load(open("configs/main.json"))


def wrap_ordinary_observations(observations, last_trade: TestOrder):
    # Wraps the observations in the format expected by the model 

    if len(observations) != cfg["observation_window"]:
        raise ValueError("The number of observations must be equal to the observation window")

    obs = observations[cfg["observed_features"]].values.flatten()
    current_price = observations["Close"].values[-1]
    if last_trade.exit_timestamp is not None:
        obs = np.append(obs, [(current_price / last_trade.entry_price - 1) * 100, True])
    else:
        obs = np.append(obs, [0, False])
    return obs


def wrap_knife_observations(observations, last_trade: TestOrder):
    # Wraps the observations in the format expected by the model 

    if len(observations) != cfg["observation_window"]:
        raise ValueError("The number of observations must be equal to the observation window")

    obs = observations[cfg["observed_features"]].values.flatten()
    current_price = observations["Close"].values[-1]
    if last_trade.exit_timestamp is not None:
        profitloss = calc_profit(last_trade.entry_price, current_price) - 0.1
        obs = np.append(obs, [profitloss, last_trade.last_action])
    else:
        obs = np.append(obs, [0, last_trade.last_action])

    return obs


def main():
    # Load the model
    model = PPO.load(cfg["model_path"])
    
    last_trade = TestOrder("BTCUSD", None, None, None, cfg["strategy"])

    if cfg["receiver_endpoint"]:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(cfg["receiver_endpoint"])

    # Define the next prediction time
    # Round the current time to the nearest 15 seconds
    next_prediction_time = np.ceil(time.time() / cfg["prediction_frequency"]) * cfg["prediction_frequency"]
    
    while True:
        # Get the current time
        current_time = time.time()

        # If the current time is greater than the next prediction time, make a prediction
        if current_time >= next_prediction_time:
            
            observations = get_last_data(int(cfg['observation_window']*cfg['prediction_frequency']/60), "15s")

            # Wrap the observations
            if cfg["strategy"] == "No strategy":
                obs = wrap_ordinary_observations(observations, last_trade)
            elif cfg["strategy"] == "Knife":
                obs = wrap_knife_observations(observations, last_trade)

            # Make a prediction
            action, _ = model.predict(obs, deterministic=True)

            # Save the last trade
            if action == 1:
                last_trade = TestOrder("BTCUSD", observations["Close"].values[-1], observations["unix"].values[-1], "15s", cfg["strategy"])
            elif action == 2:
                last_trade.exit_price = observations["Close"].values[-1]
                last_trade.exit_timestamp = observations["unix"].values[-1]
                last_trade.profit = calc_profit(last_trade.entry_price, last_trade.exit_price) - 0.1
            
            last_trade.last_action = action

            # Send the action to the endpoint
            # For now, we will just print the action
            if cfg["receiver_endpoint"]:
                # Send last_trade to endpoint
                socket.send_json(last_trade.__dict__)
                response = socket.recv()
                print(f"Action: {last_trade.last_action}, Response: {response}")
            else:
                print(f"Action: {last_trade.last_action}")

            # Update the next prediction time
            next_prediction_time = np.ceil(time.time() / cfg["prediction_frequency"]) * cfg["prediction_frequency"]

        # Sleep for 1 second
        time.sleep(0.1)
            
    

if __name__ == "__main__":
    main()