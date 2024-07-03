import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .Order import TestOrder
from .reward_functions import testOrder_reward
from .reward_functions import calc_profit
from typing import List
from RL_utils.data_handling import EpisodeManager

class TrainEnv(gym.Env):
    
    def __init__(self, data, window_size, training_period=1000, transaction_cost_pct=0.001, test_mode=False, balance=10000, tokens_held=0):
        super(TrainEnv, self).__init__()
        
        # Initialize parameters
        self.test_mode = test_mode
        self.data = data
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        # traveler index of row in dataframe. first is random
        self.current_step = np.random.randint(self.window_size, len(self.data) - 2) if not self.test_mode else window_size
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
        self.bought = np.inf # to calculate reward selling price / buying price

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell

        # TODO: fix, if flatten applied
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, len(data.iloc[0]) - 2), 
            dtype=np.float32
        )

        # Initialize past actions tracker
        self.past_actions = [] # (timestamp, action, price, balance, tokens_held)

    def reset(self, seed=None):
        self.current_step = np.random.randint(self.window_size, len(self.data) - 2) if not self.test_mode else self.window_size
        self.balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.last_time_step = min(self.current_step + self.training_period, len(self.data) - 2) if not self.test_mode else len(self.data) - 2
        self.past_actions = []
        self.action_rule_violation = False
        self.min_observed_price = np.inf
        self.min_observed_price_temp = np.inf 
        self.max_observed_price = -np.inf
        self.bought = np.inf

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
        # converting to numpy array and drop price and timestamps
        obs = frame.values[:, 2:]
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
            self.bought = self.data['Close'].iloc[self.current_step]
        elif action == 2: # Sell
            if self.tokens_held > 0:
                self.balance = self.tokens_held * current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Sell', current_price, self.balance, self.tokens_held))
            else:
                # nothing to sell
                self.action_rule_violation = True
        else: # Hold
            self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))

    def _get_reward(self):
        current_price = self.data['Close'].iloc[self.current_step]
        next_price = self.data['Close'].iloc[self.current_step + 1]
        reward = 0
        if self.balance > 0:
            # if we sold all tokens, we lost potential profit
            lost_profit = next_price / current_price - 1
            reward -= lost_profit
        elif self.tokens_held > 0:
            # if we hold tokens, we gain potential profit or loss
            observed_profit = next_price / current_price - 1
            reward += observed_profit


        reward -= self.action_rule_violation
        if self.past_actions and self.past_actions[-1][1] == 'Buy':
            reward += current_price / self.bought - 1
        return reward
    
    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Tokens held: {self.tokens_held}, Total transactions: {self.total_transactions}, Min price: {self.min_observed_price}, Max price: {self.max_observed_price}, Action rule violation: {self.action_rule_violation}')
        print(f'Past actions: {self.past_actions}')
        print('')



class SparseTrainEnv(gym.Env):
    """
    Custom gym environment for training a sparse reinforcement learning agent.

    Sparse reinforcement learning (RL) refers to a type of RL problem where the agent receives a sparse reward signal, meaning that the reward is only provided at certain specific time steps or states. 
    In contrast, in dense RL, the agent receives a reward signal at every time step or state.

    Parameters:
    - data (List[pandas.DataFrame]): List of input dataframes used for training the agent.
    - window_size (int): The size of the sliding window used for observation.
    - episod_length (int, optional): The number of time steps used for training. Default is 1000.
    - transaction_cost_pct (float, optional): The transaction cost as a percentage of the transaction value. Default is 0.001.
    - test_mode (bool, optional): Flag indicating whether the environment is in test mode. Default is False.
    """

    def __init__(
            self, 
            data: List[pd.DataFrame], 
            window_size, 
            episode_length=1000, 
            observation_window=10,
            episod_step=20, 
            episod_temp=1,
            transaction_cost_pct=0.001, 
            test_mode=False
            ):
        super(SparseTrainEnv, self).__init__()
        # Training parameters
        self.test_mode = test_mode
        self.data = data
        self.episode_manager = EpisodeManager(dataframes=data, left_indent=window_size, right_indent=2, episod_step=episod_step, observation_window=observation_window)
        self.window_size = window_size
        self.episode_length = episode_length
        self.episod_temp = episod_temp

        # Initialize the current dataframe and step
        if not self.test_mode:
            self.df_id, self.episode_start = self.episode_manager.select_episode(temperature=self.episod_temp)
            self.current_step = self.episode_start
            self.last_time_step = min(self.current_step + self.episode_length, len(self.data[self.df_id]) - 2)
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episod_temp)
            self.episode_start = window_size
            self.current_step = self.episode_start
            self.last_time_step = len(self.data[self.df_id]) - 2

        # Environment parameters
        self.transaction_cost_pct = transaction_cost_pct
        self.balance = 10000 # Initial balance
        self.entering_balance = 10000 # Initial balance
        self.tokens_held = 0 # Number of tokens held
        self.total_transactions = 0 # Total number of transactions
        self.past_actions = [] # List of past actions taken
        self.cumulative_reward = 0 # Total reward over the episode
        self.action_rule_violation = False # Flag indicating whether an action rule was violated (e.g., buying without sufficient balance)
        self.min_observed_price = np.inf # Minimum observed price for buying
        self.max_price_gap = 0 # Maximum historical price gap in percentage
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * (len(data[self.df_id].iloc[0]) - 2) + 2,), # Previous observations + current profit + holding flag
            dtype=np.float32
        )

    def reset(self, seed=None):
        """
        Reset the environment to its initial state.

        Returns:
        - observation (numpy.ndarray): The initial observation of the environment.
        - info (dict): Additional information about the reset.
        """
        if not self.test_mode:
            self.df_id, self.episode_start = self.episode_manager.select_episode(temperature=self.episod_temp)
            self.current_step = self.episode_start
            self.last_time_step = min(self.current_step + self.episode_length, len(self.data[self.df_id]) - 2)
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episod_temp)
            self.episode_start = self.window_size
            self.current_step = self.episode_start
            self.last_time_step = len(self.data[self.df_id]) - 2
        
        self.balance = 10000
        self.entering_balance = 10000
        self.tokens_held = 0
        self.total_transactions = 0
        self.past_actions = []
        self.cumulative_reward = 0
        self.action_rule_violation = False
        self.min_observed_price = np.inf
        self.max_price_gap = 0

        return self._next_observation(), {}
    
    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
        - action (int): The action to take in the environment.

        Returns:
        - observation (numpy.ndarray): The new observation of the environment.
        - reward (float): The reward received from the previous action.
        - done (bool): Flag indicating whether the episode is done.
        - info (dict): Additional information about the step.
        """
        self.action_rule_violation = False
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= self.last_time_step

        reward = self._get_reward(done)
        # Update the episode statistics if the episode is done
        if done:
            self.episode_manager.update_stats(self.df_id, self.episode_start, reward)
        self.cumulative_reward += reward

        return self._next_observation(), reward, done, False, {}
    
    def _next_observation(self):
        """
        Get the next observation of the environment.

        Returns:
        - observation (numpy.ndarray): The next observation of the environment.
        """
        frame = self.data[self.df_id].iloc[self.current_step - self.window_size + 1:self.current_step + 1]
        current_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        obs = frame.values[:, 2:].flatten()
        current_profit = self.balance + self.tokens_held * current_price
        current_profit = (current_profit / 10000 - 1)
        obs = np.append(obs, [current_profit, self.tokens_held > 0]) # Previous observations + current profit + holding flag

        if current_price < self.min_observed_price:
            self.min_observed_price = current_price
        elif current_price / self.min_observed_price - 1 > self.max_price_gap:
            self.max_price_gap = current_price / self.min_observed_price - 1

        return obs

    def _take_action(self, action):
        """
        Take the specified action in the environment.

        Parameters:
        - action (int): The action to take in the environment.
        """
        current_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        timestamp = self.data[self.df_id]['unix'].iloc[self.current_step]
                
        if action == 1:  # Buy
            if self.balance > 0:
                self.entering_balance = self.balance
                tokens_bought = self.balance / current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = tokens_bought
                self.balance = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Buy', current_price, self.balance, self.tokens_held))
            else:
                self.action_rule_violation = True
                self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))
        elif action == 2:  # Sell
            if self.tokens_held > 0:
                self.balance = self.tokens_held * current_price * (1 - self.transaction_cost_pct)
                self.tokens_held = 0
                self.total_transactions += 1
                self.past_actions.append((timestamp, 'Sell', current_price, self.balance, self.tokens_held))
            else:
                self.action_rule_violation = True
                self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))
        else:  # Hold
            self.past_actions.append((timestamp, 'Hold', current_price, self.balance, self.tokens_held))

    def _get_reward(self, done):
        """
        Calculate the reward based on the current state of the environment.

        Parameters:
        - done (bool): Flag indicating whether the episode is done.

        Returns:
        - reward (float): The reward for the current state.
        """
        if done:
            final_net_worth = self.balance + self.tokens_held * self.data[self.df_id]['Close'].iloc[self.current_step]
            reward = (final_net_worth / 10000 - 1) * 100  # Total net worth as a percentage gain
            reward -= self.max_price_gap * 100 # Use the maximum price gap as a baseline for the reward

        elif self.past_actions and self.past_actions[-1][1] == 'Sell':
            reward = (self.balance / self.entering_balance - 1) * 100  # Profit as a percentage gain
        else:
            reward = 0  # Intermediate steps do not contribute to the reward directly
        reward -= self.action_rule_violation
        return reward
    
def print_step(self):
        df_stats = self.episode_manager.stats
        # For each dataframe print the average fitness score
        print("========================================")
        print("\t\t".join([str(df_id) for df_id in df_stats.keys()]))
        print("\t\t".join([f"{df_stats[df_id]['total_fitness'] / len(df_stats[df_id]['episodes_stats']):.2f}" for df_id in df_stats.keys()]))
        

class KnifeEnv(gym.Env):
    def __init__(self, 
            data: List[pd.DataFrame], 
            window_size, 
            episode_length=1000, 
            observation_window=10,
            episode_step=20, 
            episode_temp=1,
            transaction_cost_pct=0.001, 
            test_mode=False
            ):
        super(KnifeEnv, self).__init__()
        # training parameters
        self.test_mode = test_mode
        self.data = data # list of df
        self.episode_manager = EpisodeManager(dataframes=data, left_indent=window_size, right_indent=2, episod_step=episode_step, observation_window=observation_window)
        self.window_size = window_size # how many previous rows to show to agent
        self.episode_length = episode_length
        self.episode_temp = episode_temp
        
        # Initialize the current dataframe and step
        if not self.test_mode:
            self.df_id, self.episode_start = self.episode_manager.select_episode(temperature=self.episode_temp)
            self.current_step = self.episode_start
            self.last_step = min(self.current_step + self.episode_length, len(self.data[self.df_id]) - 2)
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episode_temp)
            self.episode_start = self.window_size
            self.current_step = self.episode_start
            self.last_step = len(self.data[self.df_id]) - 2
        
        # environment parameters 
        self.trades=[]
        self.rewards=[]
        self.action_space = spaces.Discrete(3)
        self.previous_action=0 # do nothing by default
        
        self.low=np.array ([-1000,-1000,-1000,10]*self.window_size + [-1,0])
        self.high=np.array([1000,  1000, 1000, 20000]*self.window_size+[1,2])
        
        # Previous observations * window_size (unix and Close dropped) + current profit + previous_action. Dimension: (4*20+2)x1
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, 
            shape=(window_size * (len(data[self.df_id].iloc[0]) - 2) + 2,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        # after episode end, reset environment
        if not self.test_mode:
            self.df_id, self.episode_start = self.episode_manager.select_episode(temperature=self.episode_temp)
            self.current_step = self.episode_start
            self.last_step = min(self.current_step + self.episode_length, len(self.data[self.df_id]) - 2)
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episode_temp)
            self.episode_start = self.window_size
            self.current_step = self.episode_start
            self.last_step = len(self.data[self.df_id]) - 2
        
        self.trades=[]
        self.previous_action=0
        return self._next_observation(), {}
    
    def step(self, action):
        # Evaluate, if action is possible. Then get reward for episode
        self._take_action(action)
        self.current_step += 1
        # flag for reseting environment
        done = self.current_step >= self.last_step
        # Analyze action
        reward = self._get_reward(action, done)
        self.previous_action=action
        if done:
            self.episode_manager.update_stats(self.df_id, self.episode_start, reward)
            info= {'trades' : self.trades, 'rewards':self.rewards}
        else:
            info = {}
        return self._next_observation(), reward, done, False, info
    
    def _next_observation(self):
        # new data
        frame = self.data[self.df_id].iloc[self.current_step - self.window_size + 1:self.current_step + 1]
        obs = frame.values[:, [1,3,4,5]].flatten()
        current_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        
        # unrealized profit calc to pass along other observations
        # if dont have opened trade
        if len(self.trades)==0 or (len(self.trades)!=0 and self.trades[-1].exit_price!=None):
            current_profitloss=0
        # if have opened trade
        elif self.trades[-1].exit_price==None:
            current_profitloss = calc_profit(self.trades[-1].entry_price,current_price)
            self.trades[-1].profit=current_profitloss
        
        # Previous observations + unrealized profit if have opened trade + previous action
        obs = np.append(obs, [current_profitloss, self.previous_action]) 
        
        # check if values of obs are within bounds
        if not self.observation_space.contains(obs):
            out_of_bounds = (obs < self.observation_space.low) | (obs > self.observation_space.high)
            if np.any(out_of_bounds):
                print(obs[out_of_bounds])
        return obs
    
    def _take_action(self, action, symbol='BTCUSDT', period='15s',strategy='knife'):
        # update last trade info
        current_price=self.data[self.df_id]['Close'].iloc[self.current_step]
        current_timestamp=self.data[self.df_id]['unix'].iloc[self.current_step]
        
        if action==1: # buy 
            # if first order or previous order was sold
            if len(self.trades)==0 or (len(self.trades)>0 and self.trades[-1].exit_price!=None):
                self.trades.append(TestOrder(symbol, current_price, current_timestamp, period, strategy))
        elif action==2: # sell
            # if previous order only bought
            if len(self.trades)>0 and self.trades[-1].exit_price==None:
                self.trades[-1].exit_price=current_price
                self.trades[-1].exit_timestamp=current_timestamp        
        else: # do nothing
            pass
    
    def _get_reward(self, action, done):
        current_price=self.data[self.df_id]['Close'].iloc[self.current_step]
        # if have opened trade, or trade which is closed in current price
        current_trade=self.trades[-1] if len(self.trades)>0 and (self.trades[-1].exit_price==None or self.trades[-1].exit_price==current_price) else None
        
        # if episode didnt end and have opened trade
        if not done and current_trade!=None:
            
            # if action is do nothing and trade is opened, return unrealized profit
            if action==0 and current_trade.exit_price==None:
                reward=5*calc_profit(current_trade.entry_price, current_price)
                
            # if action is to buy, when trade is opened, return penalty
            elif action==1 and current_trade.exit_price==None:
                reward=-0.3
                
            # if action is to sell, when trade is closed in current step, return reward
            elif action==2 and current_trade.exit_price==current_price:
                reward=testOrder_reward(current_trade)
                
        # if episode didnt end and not have opened trade
        elif not done and current_trade==None:
            reward=0
        
        # Penalties for same action=(1,2) in row
        elif self.previous_action==action and action!=0:
            reward= -0.3
        elif self.previous_action==action and action==0:
            reward= 0
        # Penalty for sell, if not have opened trade
        elif current_trade==None and action==2:
            reward=-0.3
            
        # If episode end, return reward for all trades
        elif done:
            reward=sum(testOrder_reward(testOrder) for testOrder in self.trades)
            # penalize for no trades
            if reward==0:
                reward=-1
        
        self.rewards.append(reward)
        return reward
    
    def render(self):
        # Print current info
        # print(self._get_reward())
        pass