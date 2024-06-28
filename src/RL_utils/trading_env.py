import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TrainEnv(gym.Env):
    """Custom Environment that follows gym interface.

    Parameters:
    - data (pandas.DataFrame): The input data for the environment.
    - window_size (int): The size of the observation window.
    - transaction_cost_pct (float, optional): The transaction cost percentage. Defaults to 0.001.

    Attributes:
    - metadata (dict): Metadata for the environment.
    - action_space (gym.spaces.Discrete): The action space for the environment.
    - observation_space (gym.spaces.Box): The observation space for the environment.
    - current_step (int): The current step in the environment.
    - balance (float): The current balance in the environment.
    - shares_held (float): The number of shares held in the environment.
    - total_shares_sold (float): The total number of shares sold in the environment.
    - total_sales_value (float): The total sales value in the environment.
    - past_actions (list): A list of past actions taken in the environment.

    Methods:
    - reset(): Resets the environment to its initial state.
    - step(action): Takes a step in the environment based on the given action.
    - _next_observation(): Returns the next observation in the environment.
    - _take_action(action): Takes the given action in the environment.
    - _calculate_reward(): Calculates the reward for the current step in the environment.
    - render(): Renders the current state of the environment.
    - close(): Closes the environment.

    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, data, window_size, transaction_cost_pct=0.001):
        super(TrainEnv, self).__init__()
        
        # Initialize parameters
        self.data = data
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        self.current_step = np.random.randint(self.window_size, len(self.data) - 1)
        self.balance = 10000  # Initial balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, len(data.columns)), 
            dtype=np.float32
        )
        
        # Initialize past actions tracker
        self.past_actions = []

    def reset(self):
        """Resets the environment to its initial state.

        Returns:
        - observation (numpy.ndarray): The initial observation.
        - info (dict): Additional information about the reset.

        """
        self.current_step = np.random.randint(self.window_size, len(self.data) - 1)
        self.balance = 10000
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.past_actions = []
        
        return self._next_observation(), {}

    def step(self, action):
        """Takes a step in the environment based on the given action.

        Parameters:
        - action (int): The action to take.

        Returns:
        - observation (numpy.ndarray): The next observation.
        - reward (float): The reward for the current step.
        - terminated (bool): Whether the episode is terminated.
        - info (dict): Additional information about the step.

        """
        self._take_action(action)
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            terminated = True
        else:
            terminated = False
        
        reward = self._calculate_reward()
        obs = self._next_observation()

        return obs, reward, terminated, False, {}

    def _next_observation(self):
        """Returns the next observation in the environment.

        Returns:
        - observation (numpy.ndarray): The next observation.

        """
        frame = np.array(self.data.iloc[self.current_step - self.window_size + 1 : self.current_step + 1])
        past_action_frame = np.array([self.balance, self.shares_held, self.total_sales_value])

        return np.hstack((frame, past_action_frame.reshape(-1, 1)))

    def _take_action(self, action):
        """Takes the given action in the environment.

        Parameters:
        - action (int): The action to take.

        """
        current_price = self.data.iloc[self.current_step]['Close']
        
        if action == 1:  # Buy
            total_possible = self.balance // current_price
            self.shares_held += total_possible
            self.balance -= total_possible * current_price * (1 + self.transaction_cost_pct)
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price * (1 - self.transaction_cost_pct)
            self.total_shares_sold += self.shares_held
            self.total_sales_value += self.shares_held * current_price
            self.shares_held = 0
        self.past_actions.append(action)

    def _calculate_reward(self):
        """Calculates the reward for the current step in the environment.

        Returns:
        - reward (float): The reward for the current step.

        """
        net_worth = self.balance + self.shares_held * self.data.iloc[self.current_step]['Close']
        reward = net_worth - 10000
        return reward

    def render(self):
        """Renders the current state of the environment."""
        profit = self.balance + self.shares_held * self.data.iloc[self.current_step]['Close'] - 10000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total sales: {self.total_sales_value}')
        print(f'Net Profit: {profit}')

    def close(self):
        """Closes the environment."""
        pass
