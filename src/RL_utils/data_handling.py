import numpy as np
import pandas as pd
from typing import List, Tuple

class EpisodeManager:
    """
    A class that manages episodes and statistics for reinforcement learning.\n
    Stores the statistics about agent's fitness to given dataframes and episodes.
    Fitness is calculated as the ratio of the exponentially weighted moving average (EWMA) score to the observed range of scores.

    Args:
        dataframes (list): A list of dataframes.
        left_indent (int): The left indent value for episode statistics.
        right_indent (int): The right indent value for episode statistics.
        episod_step (int): The step for episode window.
        alpha (float): The alpha value for calculating the exponentially weighted moving average (EWMA) score.

    Attributes:
        left_indent (int): The left indent value for episode statistics.
        right_indent (int): The right indent value for episode statistics.
        episod_step (int): The step size for episode statistics.
        alpha (float): The alpha value for calculating the EWMA score.
        stats (dict): A dictionary that stores the statistics for each dataframe and episode.

    Methods:
        update_stats(df_id, episode_start, episode_score):
            Updates the statistics for a given dataframe and episode.
        select_dataframe(temperature=1):
            Selects a dataframe based on the total fitness scores.
        select_episode(temperature=1):
            Selects an episode based on the fitness scores within a dataframe.
    """

    def __init__(self, dataframes: List[pd.DataFrame], left_indent=20, right_indent=2, episod_step=1, alpha=0.1):
        self.left_indent = left_indent
        self.right_indent = right_indent
        self.episod_step = episod_step
        self.alpha = alpha
        
        self.stats = {
            df_id: {
                "total_fitness": 0,
                "episodes_stats": {
                    episode_start: {
                        "max_score": -np.inf,
                        "min_score": np.inf,
                        "EWMA_score": 0,
                        "fitness": 0,
                    } for episode_start in range(self.left_indent, len(dataframes[df_id]) - self.right_indent, self.episod_step)
                }
            } for df_id in range(len(dataframes))
        }

    def update_stats(self, df_id:int, episode_start:int, episode_score:float):
        """
        Updates the statistics for a given dataframe and episode.

        Args:
            df_id (int): The ID of the dataframe.
            episode_start (int): The starting index of the episode.
            episode_score (float): The score of the episode.
        """
        # Load the stats for the current dataframe and episode
        df_stats = self.stats[df_id]
        episode_stats = df_stats["episodes_stats"][episode_start]

        # Update the episode stats
        episode_stats["max_score"] = max(episode_score, episode_stats["max_score"])
        episode_stats["min_score"] = min(episode_score, episode_stats["min_score"])
        episode_stats["EWMA_score"] = self.alpha * episode_score + (1 - self.alpha) * episode_stats["EWMA_score"]
        prev_fitness = episode_stats["fitness"]
        
        # Update the fitness score
        if episode_stats["max_score"] != episode_stats["min_score"]:  # Avoid division by zero
            episode_stats["fitness"] = (episode_stats["EWMA_score"] - episode_stats["min_score"]) / (episode_stats["max_score"] - episode_stats["min_score"])

        # Update the total fitness for the dataframe
        df_stats["total_fitness"] += episode_stats["fitness"] - prev_fitness

    def select_dataframe(self, temperature=1.0):
        """
        Selects a dataframe based on the total fitness scores.

        Args:
            temperature (float): The temperature parameter for softmax selection.

        Returns:
            int: The ID of the selected dataframe.
        """
        # Compute the probabilities for each dataframe
        probabilities = np.array([np.exp(df_stats["total_fitness"] / len(df_stats["episodes_stats"]) / temperature) for df_stats in self.stats.values()])
        probabilities /= np.sum(probabilities)

        # Select a dataframe
        return np.random.choice(list(self.stats.keys()), p=probabilities)
    
    def select_episode(self, temperature=1.0):
        """
        Selects an episode based on the fitness scores within a dataframe.

        Args:
            temperature (float): The temperature parameter for softmax selection.

        Returns:
            tuple: A tuple containing the ID of the selected dataframe and the starting index of the selected episode.
        """
        # Select a dataframe
        df_id = self.select_dataframe(temperature)
        df_stats = self.stats[df_id]

        # Compute the probabilities for each episode
        probabilities = np.array([np.exp(episode_stats["fitness"] / temperature) for episode_stats in df_stats["episodes_stats"].values()])
        probabilities /= np.sum(probabilities)

        # Select an episode
        episode_start = np.random.choice(list(df_stats["episodes_stats"].keys()), p=probabilities)
        return df_id, episode_start
