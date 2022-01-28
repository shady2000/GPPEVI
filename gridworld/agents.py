import numpy as np
import pickle
from env import ACTIONS, GridWorld
environment = GridWorld()

class Q_Agent():
    def __init__(self, epsilon=0.05, alpha=0.1, gamma=1):
        self.q_table = dict() # Store all Q-values in dictionary of dictionaries 
        for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(environment.width):
                self.q_table[(x,y)] = {0:0, 1:0, 2:0, 3:0} # Populate sub-dictionary with zero values for possible moves

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def choose_action(self, available_actions, h):
        if np.random.uniform(0,1) < self.epsilon:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
        
        return action
    
    def learn(self, old_state, reward, new_state, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]
        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)

class RandomAgent():        
    def choose_action(self, state, h):
        return np.random.choice(ACTIONS)

class PEVIAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)

    def choose_action(self, state, h):
        prob_raw = self.policy_all[state][:,h] 
        prob = prob_raw/np.sum(prob_raw)
        action = np.random.choice(ACTIONS, p=prob)
        return action

class GPAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)
    def choose_action(self, state, h):
        policy_raw = self.policy_all[state[0],state[1],state[2], :, h]
        policy = policy_raw/policy_raw.sum() 
        action = np.random.choice(ACTIONS, p=policy)
        return np.array([action])

class FQIAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)
    def choose_action(self, state, h):
        policy_raw = self.policy_all[state[0],state[1],state[2], :]
        policy = policy_raw/policy_raw.sum() 
        action = np.random.choice(ACTIONS, p=policy)
        return np.array([action])

class PQL_BCQ_Agent(): 
    def __init__(self, policy_log): 
        with open(policy_log, 'rb') as inp:
            self.policy = pickle.load(inp)
    def choose_action(self, state, h):
        return self.policy.select_action(np.array(state))