import numpy as np

class EpisodeManager:
    def __init__(self, dataframes, left_indent=20, right_indent=2, episod_step=1, alpha=0.1):
        self.dataframes = dataframes
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

    def update_stats(self, df_id, episode_start, episode_score):
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

    def select_dataframe(self, temperature=1):
        # Compute the probabilities for each dataframe
        probabilities = np.array([np.exp(df_stats["total_fitness"] / len(df_stats["episodes_stats"]) / temperature) for df_stats in self.stats.values()])
        probabilities /= np.sum(probabilities)

        # Select a dataframe
        return np.random.choice(list(self.stats.keys()), p=probabilities)
    
    def select_episode(self, temperature=1):
        # Select a dataframe
        df_id = self.select_dataframe(temperature)
        df_stats = self.stats[df_id]

        # Compute the probabilities for each episode
        probabilities = np.array([np.exp(episode_stats["fitness"] / temperature) for episode_stats in df_stats["episodes_stats"].values()])
        probabilities /= np.sum(probabilities)

        # Select an episode
        episode_start = np.random.choice(list(df_stats["episodes_stats"].keys()), p=probabilities)
        return df_id, episode_start
