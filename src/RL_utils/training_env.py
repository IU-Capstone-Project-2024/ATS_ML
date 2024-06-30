import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TrainEnv(gym.Env):
    
    def __init__(self, data, window_size, training_period=1000, transaction_cost_pct=0.001, test_mode=False, balance=10000, tokens_held=0):
        super(TrainEnv, self).__init__()
        
        # Initialize parameters
        self.test_mode = test_mode
        self.data = data
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        # traveler index of row in dataframe. first is random
        self.current_step = np.random.randint(self.window_size, len(self.data) - 2) if not self.test_mode else 0 
        self.balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.training_period = training_period  # how many rows to see
        self.last_time_step = min(self.current_step + self.training_period, len(self.data) - 2) if not self.test_mode else len(self.data) - 2
        self.action_rule_violation = False # model should take action, but dont have needed asset
        # Used for calculating one potential best trade
        self.min_observed_price = np.inf # price to buy
        self.min_observed_price_temp = np.inf # while price decline, save potential buy price
        self.max_observed_price = -np.inf # price to sell
        self.bought = False

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell

        # TODO: fix, if flatten applied
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, len(data.iloc[0])), 
            dtype=np.float32
        )

        # Initialize past actions tracker
        self.past_actions = [] # (timestamp, action, price, balance, tokens_held)

    def reset(self, seed=None):
        self.current_step = np.random.randint(self.window_size, len(self.data) - 2) if not self.test_mode else 0
        self.balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.last_time_step = min(self.current_step + self.training_period, len(self.data) - 2) if not self.test_mode else len(self.data) - 2
        self.past_actions = []
        self.action_rule_violation = False
        self.min_observed_price = np.inf
        self.min_observed_price_temp = np.inf 
        self.max_observed_price = -np.inf
        self.bought = False

        return self._next_observation(), {}
    
    def step(self, action):
        self.action_rule_violation = False
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= self.last_time_step:
            # traveled until end
            done = True
        else:
            done = False

        return self._next_observation(), self._get_reward(), done, False, {}
    
    def _next_observation(self):
        # get next timestamp data + window_size previous
        frame = self.data.iloc[self.current_step - self.window_size + 1:self.current_step + 1]
        # creating empty numpy array
        obs = np.ndarray(shape=(self.window_size, len(frame.iloc[0])), dtype=np.float32)
        # filling with values
        # TODO: apply flatten, if needed, append self.bought
        obs = frame.to_numpy()
        return obs

    def _take_action(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        timestamp = self.data['unix'].iloc[self.current_step]
        
        # update potential best trade
        # if current_price > self.max_observed_price:
        #     self.max_observed_price = current_price
        #     self.min_observed_price = self.min_observed_price_temp
        # still searching for best entry
        # elif current_price < self.min_observed_price_temp:
        #     self.min_observed_price_temp = current_price

        
        if action == 1: # Buy
            if self.balance > 0:
                tokens_bought = self.balance / current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = tokens_bought
                self.balance = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Buy', current_price, self.balance, self.tokens_held))
            else:
                # already bought
                self.action_rule_violation = True
            self.bought = True
        elif action == 2: # Sell
            if self.tokens_held > 0:
                self.balance = self.tokens_held * current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Sell', current_price, self.balance, self.tokens_held))
            else:
                # nothing to sell
                self.action_rule_violation = True
            self.bought = False
        else: # Hold
            self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))

    def _get_reward(self):
        current_price = self.data['Close'].iloc[self.current_step]
        next_price = self.data['Close'].iloc[self.current_step + 1]
        current_net_worth = self.balance + self.tokens_held * current_price
        next_net_worth = self.balance + self.tokens_held * next_price
        reward = next_net_worth / current_net_worth
        reward -= self.action_rule_violation
        return reward
    
    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Tokens held: {self.tokens_held}, Total transactions: {self.total_transactions}, Min price: {self.min_observed_price}, Max price: {self.max_observed_price}, Action rule violation: {self.action_rule_violation}')
        print(f'Past actions: {self.past_actions}')
        print('')
