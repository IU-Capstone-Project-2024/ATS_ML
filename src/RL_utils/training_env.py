import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TrainEnv(gym.Env):
    
    def __init__(self, data, window_size, training_period, transaction_cost_pct=0.001, balance=10000, tokens_held=0):
        super(TrainEnv, self).__init__()
        
        # Initialize parameters
        self.data = data
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        self.current_step = np.random.randint(self.window_size, len(self.data) - 1)
        self.balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.training_period = training_period
        self.last_time_step = min(self.current_step + self.training_period, len(self.data) - 1)
        self.action_rule_violation = False
        self.min_observed_price = np.inf
        self.min_observed_price_temp = np.inf
        self.max_observed_price = -np.inf

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, 4), 
            dtype=np.float32
        )

        # Initialize past actions tracker
        self.past_actions = [] # (timestamp, action, price, balance, tokens_held)

    def reset(self, seed=None):
        self.current_step = np.random.randint(self.window_size, len(self.data) - 1)
        self.balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.last_time_step = min(self.current_step + self.training_period, len(self.data) - 1)
        self.past_actions = []
        self.action_rule_violation = False
        self.min_observed_price = np.inf
        self.min_observed_price_temp = np.inf
        self.max_observed_price = -np.inf

        return self._next_observation(), {}
    
    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= self.last_time_step:
            done = True
        else:
            done = False

        return self._next_observation(), self._get_reward(), done, False, {}
    
    def _next_observation(self):
        frame = self.data.iloc[self.current_step - self.window_size + 1:self.current_step + 1]
        obs = np.ndarray(shape=(self.window_size, 4), dtype=np.float32)
        obs[:, 0] = frame["Close"] / frame["Open"]
        obs[:, 1] = frame["High"] / frame["Low"]
        obs[:, 2] = frame["Volume"]
        obs[:, 3] = frame["Number of Trades"]
        return obs

        
    
    def _take_action(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        timestamp = self.data.index[self.current_step]
        if current_price > self.max_observed_price:
            self.max_observed_price = current_price
            self.min_observed_price = self.min_observed_price_temp
        elif current_price < self.min_observed_price_temp:
            self.min_observed_price_temp = current_price

        
        if action == 1: # Buy
            if self.balance > 0:
                tokens_bought = self.balance / current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = tokens_bought
                self.balance = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Buy', current_price, self.balance, self.tokens_held))
            else:
                self.action_rule_violation = True
        elif action == 2: # Sell
            if self.tokens_held > 0:
                self.balance = self.tokens_held * current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Sell', current_price, self.balance, self.tokens_held))
            else:
                self.action_rule_violation = True
        else: # Hold
            self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))

    def _get_reward(self):
        current_price = self.data['Close'].iloc[self.current_step]
        net_worth = self.balance + self.tokens_held * current_price
        if not self.min_observed_price == np.inf:
            reward = (net_worth - 10000) / (self.max_observed_price - self.min_observed_price)
        else:
            reward = net_worth - 10000
        return reward
    
    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Tokens held: {self.tokens_held}, Total transactions: {self.total_transactions}, Min price: {self.min_observed_price}, Max price: {self.max_observed_price}, Action rule violation: {self.action_rule_violation}')
        print(f'Past actions: {self.past_actions}')
        print('')


