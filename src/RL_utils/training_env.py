import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .Order import TestOrder
from .reward_functions import testOrder_reward
from .reward_functions import calc_profit
from typing import List
from RL_utils.data_handling import EpisodeManager


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

        # Reward parameters
        self.short_distance_weight=-1.8 # c
        self.long_distance_weight=0.1 # a
        self.least_penalized_distance=10 # x_min
        self.min_penalty_scale = 1 # b

        self.zero_distance_penalty = self.calculate_zero_distance_penalty()       

        
        self.distance_weight_diff = self.long_distance_weight - self.short_distance_weight
        self.distance_weight_sum = self.long_distance_weight + self.short_distance_weight
        self.distance_const_diff = self.min_penalty_scale - self.zero_distance_penalty
        self.distance_const_sum = self.min_penalty_scale + self.zero_distance_penalty 

        self.transaction_time_penalty = 0.3
        self.transaction_time_const = -0.6


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
        self.entering_price = self.data[self.df_id]['Close'].iloc[self.current_step] # Price at which the tokens were bought
        self.entering_step = self.current_step # Step at which the episode started
        self.exit_balance = 10000 # Balance after selling the tokens
        self.exit_price = self.data[self.df_id]['Close'].iloc[self.current_step]  # Price at which the tokens were sold
        self.exit_step = self.current_step # Step at which the tokens were sold
        self.tokens_held = 0 # Number of tokens held
        self.total_transactions = 0 # Total number of transactions
        self.last_transaction_step = None # Step at which the last transaction was made
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
        self.entering_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        self.entering_step = self.current_step
        self.exit_balance = 10000
        self.exit_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        self.exit_step = self.current_step
        self.tokens_held = 0
        self.total_transactions = 0
        self.last_transaction_step = None
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
        self.cumulative_reward += reward
        # Update the episode statistics if the episode is done
        if done:
            self.episode_manager.update_stats(self.df_id, self.episode_start, reward)

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
        if self.tokens_held > 0:
            current_profit = (current_price / self.data[self.df_id]['Close'].iloc[self.entering_step] - 1) * 100
        else:
            current_profit = 0
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
                self.entering_price = current_price 
                self.entering_step = self.current_step
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
                self.exit_balance = self.balance
                self.exit_price = current_price
                self.exit_step = self.current_step
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

        current_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        if done:
            final_net_worth = self.balance + self.tokens_held * current_price
            reward = (final_net_worth / 10000 - 1) * 100  # Total profit for the episode as a percentage gain
            reward -= self.max_price_gap * 100 # Use the maximum observed price gap as a baseline for the reward
        elif self.current_step == self.exit_step: # Just sold the tokens
            reward = (self.exit_balance / self.entering_balance - 1) * 100  # Profit for the closed trade as a percentage gain
            if self.last_transaction_step is not None:
                time_since_last_transaction = self.current_step - self.last_transaction_step
                reward += self._calculate_transaction_time_penalty(time_since_last_transaction)
            self.last_transaction_step = self.current_step
        elif self.current_step == self.entering_step: # Just bought the tokens
            reward = 0  # No reward for buying
            if self.last_transaction_step is not None:
                time_since_last_transaction = self.current_step - self.last_transaction_step
                reward += self._calculate_transaction_time_penalty(time_since_last_transaction)
        elif self.tokens_held > 0 and self.current_step > self.entering_step: # Holding the tokens
            running_gain = (current_price / self.entering_price - 1) * 100 # Running profit from open trade as a percentage gain
            steps_distance = self.current_step - self.entering_step # Number of steps since the trade was opened
            distance_multiplier = self._distance_penalty_multiplier(steps_distance)
            reward = running_gain * distance_multiplier # Running profit adjusted by the distance penalty
        elif self.balance > 0 and self.current_step > self.exit_step: # Sold tokens some time ago
            regret = -(current_price / self.exit_price - 1) * 100 # Regret for not holding the tokens as a percentage loss
            steps_distance = self.current_step - self.exit_step # Number of steps since the trade was closed
            distance_multiplier = self._distance_penalty_multiplier(steps_distance)
            reward = regret * distance_multiplier # Regret adjusted by the distance penalty
        else:
            reward = 0  # Intermediate steps do not contribute to the reward directly
        reward -= self.action_rule_violation
        return reward
    
    def calculate_zero_distance_penalty(self):
        """
        Calculate the penalty for zero distance (d)
        """
        ### d = b + x_min(a - c) + sqrt(-ac(a + c)^2) / ac
        distance_weight_diff = self.long_distance_weight - self.short_distance_weight
        distance_weight_sum = self.long_distance_weight + self.short_distance_weight
        distance_weight_product = self.long_distance_weight * self.short_distance_weight
        zero_distance_penalty = self.min_penalty_scale + self.least_penalized_distance * distance_weight_diff
        zero_distance_penalty += np.sqrt(-distance_weight_product * np.square(distance_weight_sum)) / distance_weight_product
        return zero_distance_penalty


    
    def _distance_penalty_multiplier(self, distance):
        """
        This function uses hyperbolic function to calculate the penalty multiplier for reward.
        Hyperbolic function is represented as (y - ax - b)(y - cx - d) = 1
        y = distance_penalty_multiplier

        y - ax - b is a long distance penalty scaling asymptote. Used to penalize the agent for long inactive periods.
        y - cx - d is a short distance penalty scaling asymptote. Used to penalize the agent for localy non-optimal actions.
        
        To calculate y we use the following formula:
        
        y = 1/2 * ((a + c)x + (b + d) + sqrt(((a - c)x + (b - d))^2 + 4)"""
        distance_penalty_multiplier = ((self.distance_weight_sum * distance) + self.distance_const_sum)
        distance_penalty_multiplier += np.sqrt(np.square(self.distance_weight_diff * distance + self.distance_const_diff) + 4)
        distance_penalty_multiplier /= 2
        return distance_penalty_multiplier
    
    def _calculate_transaction_time_penalty(self, elapsed_time):
        """
        Calculate the penalty for transaction time.
        This function uses hyperbolic function to calculate the penalty multiplier for reward.
        One asymptote y = 0
        Second asymptote y = ax + b

        Penalty is calculated as:
        y = 1/2 * (ax + b - sqrt((ax + b)^2 + 4))"""
        transaction_time_penalty = self.transaction_time_const + self.transaction_time_penalty * elapsed_time
        transaction_time_penalty -= np.sqrt(np.square(self.transaction_time_penalty * elapsed_time + self.transaction_time_const) + 4)
        transaction_time_penalty /= 2
        return transaction_time_penalty

    
def print_step(self):
        df_stats = self.episode_manager.stats
        # For each dataframe print the average fitness score
        print("========================================")
        print("\t\t".join([str(df_id) for df_id in df_stats.keys()]))
        print("\t\t".join([f"{df_stats[df_id]['total_fitness'] / len(df_stats[df_id]['episodes_stats']):.2f}" for df_id in df_stats.keys()]))
        

class KnifeEnv(gym.Env):
    # specified environment for training in high volatility periods
    def __init__(self, 
            data: List[pd.DataFrame], 
            window_size, 
            episode_length=1000, 
            observation_window=10,
            episode_step=20, 
            episode_temp=1,
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
        self.df_count=[0 for _ in range (len(self.data))]
        if not self.test_mode:
            self.df_id, self.episode_start = self.episode_manager.select_episode(temperature=self.episode_temp)
            self.current_step = self.episode_start
            self.last_step = min(self.current_step + self.episode_length, len(self.data[self.df_id]) - 2)
            self.df_count[self.df_id]+=1
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episode_temp)
            self.episode_start = self.window_size
            self.current_step = self.episode_start
            self.last_step = len(self.data[self.df_id]) - 2
            self.df_count[self.df_id]+=1
        
        # environment parameters 
        self.trades=[]
        self.rewards=[]
        self.action_space = spaces.Discrete(3)
        self.previous_action=0 # do nothing by default
        
        self.low=np.array ([-1000,-1000,-1000,10]*self.window_size +   [-3.5,0])
        self.high=np.array([1000,  1000, 1000, 20000]*self.window_size+[2,2]) #TODO: change to 2.5
        
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
            self.df_count[self.df_id]+=1
        else:
            self.df_id = self.episode_manager.select_dataframe(temperature=self.episode_temp)
            self.episode_start = self.window_size
            self.current_step = self.episode_start
            self.last_step = len(self.data[self.df_id]) - 2
            self.df_count[self.df_id]+=1
        
        self.trades=[]
        self.rewards=[]
        self.previous_action=0
        return self._next_observation(), {}
    
    def step(self, action):
        # Evaluate, if action is possible. Then get reward for episode
        self._take_action(action)
        # flag for reseting environment
        done = self.current_step >= self.last_step
        # Analyze action
        reward = self._get_reward(action, done)
        
        self.previous_action=action
        self.current_step += 1

        if done:
            self.episode_manager.update_stats(self.df_id, self.episode_start, reward)
            info= {'trades' : self.trades, 'rewards':self.rewards, 'df_count':self.df_count}
            self.render()
        else:
            info = {}
        return self._next_observation(), reward, done, False, info
    
    def _next_observation(self):
        # new data
        frame = self.data[self.df_id].iloc[self.current_step - self.window_size:self.current_step]
        obs = frame.values[:, [1,3,4,5]].flatten()
        current_price = self.data[self.df_id]['Close'].iloc[self.current_step]
        
        # unrealized profit calc to pass along other observations
        # if dont have opened trade
        if len(self.trades)==0 or (len(self.trades)!=0 and self.trades[-1].exit_price!=None):
            current_profitloss=0
        # if have opened trade
        elif self.trades[-1].exit_price==None:
            current_profitloss = calc_profit(self.trades[-1].entry_price,current_price)-0.1
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
        current_timestamp=self.data[self.df_id]['unix'].iloc[self.current_step]
        # if have opened trade, or trade which is closed in current price
        current_trade=self.trades[-1] if len(self.trades)>0 and (self.trades[-1].exit_price==None or self.trades[-1].exit_price==current_price) else None
        
        # if episode didnt end and have opened trade
        if not done and current_trade!=None:
            
            # if action is do nothing and have opened trade   or just bought and trade is opened, return unrealized profit
            if (action==0 and current_trade.exit_price==None) or (action==1 and current_trade.entry_timestamp==current_timestamp):
                reward=sum(testOrder_reward(testOrder, self.data[self.df_id], current_timestamp) for testOrder in self.trades)*pow(len(self.trades),0.5)
                
            # if action is to buy and have opened trade, when trade is opened not in current timestamp, return penalty. Penalty should be harder than average loss
            elif action==1 and current_trade.exit_price==None and current_trade.entry_timestamp!=current_timestamp:
                reward=-50
                
            # if action is to sell, when trade is closed in current step, return reward.
            # Not closing trade with little loss may hurt harder than not closing till end 
            elif action==2 and current_trade.exit_price==current_price:
                reward=sum(testOrder_reward(testOrder, self.data[self.df_id], current_timestamp) for testOrder in self.trades)*pow(len(self.trades),0.5)
        
        # Penalties
        # if episode didnt end and not have opened trade
        elif not done and current_trade==None and action==0:
            if len(self.trades)>0:
                time_since_last_trade=(current_timestamp-self.trades[-1].exit_timestamp)/1000 
            else:
                start_episode_timestamp=self.data[self.df_id]['unix'].iloc[self.episode_start]
                time_since_last_trade=(current_timestamp-start_episode_timestamp)/1000
            # Gradually increase penalty
            reward= (time_since_last_trade/1200)*(-50)
        
        # Sell, if not have opened trade
        elif current_trade==None and action==2:
            reward=-50

        # Same action=(1,2) in row
        elif not done and self.previous_action==action and action!=0:
            reward= -50

        # If episode end, return reward for all trades
        if done:
            # cumulative reward for all actions * amount actions
            reward=sum(testOrder_reward(testOrder, self.data[self.df_id], current_timestamp) for testOrder in self.trades)*pow(len(self.trades),0.5)
            # penalty for no trades = bad episode
            if reward==0:
                reward=-20
        
        self.rewards.append(reward)
        return reward
    
    def render(self):
        # Print current info
        print(self.df_id, 'df% ', self.df_count[self.df_id]/sum(self.df_count),' #trades ',len(self.trades),' episode reward ',self._get_reward(self.previous_action,True))
        pass