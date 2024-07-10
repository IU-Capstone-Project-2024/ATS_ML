import os
import json
import pandas as pd
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.abspath(__file__))
env1_stats_path = os.path.join(script_path, "../logs/environments/train_env1_stats.json")
env2_stats_path = os.path.join(script_path, "../logs/environments/train_env2_stats.json")

with open(env1_stats_path, "r") as f:
    env1_stats = json.load(f)
    # For each dataframe plot episode timelines and max and min scores
    plt.figure(figsize=(30,30))
    for df_id, df_stats in env1_stats.items():
        plt.subplot(7, 2, int(df_id) + 1)
        plt.title(f"Dataframe {df_id} average_fit: {df_stats['total_fitness'] / len(df_stats['episodes_stats'])}")
        plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["max_score"] for episode_stats in df_stats["episodes_stats"].values()], label="Max Score")
        plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["min_score"] for episode_stats in df_stats["episodes_stats"].values()], label="Min Score")
        # plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["fitness"] for episode_stats in df_stats["episodes_stats"].values()], label="Fitness")
        plt.legend()
    plt.show()

with open(env2_stats_path, "r") as f:
    env2_stats = json.load(f)
    # For each dataframe plot episode timelines and max and min scores
    plt.figure(figsize=(30,30))
    for df_id, df_stats in env2_stats.items():
        plt.subplot(7, 2, int(df_id) + 1)
        plt.title(f"Dataframe {df_id} average_fit: {df_stats['total_fitness'] / len(df_stats['episodes_stats'])}")
        plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["max_score"] for episode_stats in df_stats["episodes_stats"].values()], label="Max Score")
        plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["min_score"] for episode_stats in df_stats["episodes_stats"].values()], label="Min Score")
        # plt.plot(list(df_stats["episodes_stats"].keys()), [episode_stats["fitness"] for episode_stats in df_stats["episodes_stats"].values()], label="Fitness")
        plt.legend()
    plt.show()
        