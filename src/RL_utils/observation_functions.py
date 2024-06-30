import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

def raw_observation_preprocessor(observations:pd.DataFrame):
    """
    Preprocesses raw observations to be used in the model.
    Takes a list of observations and returns a numpy array.
    Expected columns: 
    - Open Time,
    - Open,
    - High,
    - Low,
    - Close,
    - Volume,
    - Close Time,
    - Quote Asset Volume,
    - Number of Trades,
    - Taker Buy Base Asset Volume,
    - Taker Buy Quote Asset Volume,
    - Ignore

    Returns:
    - Price change percentage,
    - Volume change percentage,
    - Number of trades change percentage,
    - Taker buy base asset volume change percentage,

    """
    price_change = observations["Close"] / observations["Open"] - 1
    high_low_diff = observations["High"] / observations["Low"] - 1
    trades_change = observations["Number of Trades"] / observations["Number of Trades"].shift(1) - 1
    