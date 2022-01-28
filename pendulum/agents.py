import gym
import numpy as np
import utils
import pickle

env = gym.make(utils.env_name)

class RandomAgent(): 
    def choose_action(self, state, h):
           return env.action_space.sample()

class PEVIAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)
    def choose_action(self, state, h):
        state = utils.obs2bin(state)
        policy_raw = self.policy_all[state[0],state[1],state[2], :, h]
        policy = policy_raw/policy_raw.sum() 
        action = np.random.choice(np.linspace(-2, 2, utils.num_act_bins), p=policy)
        return np.array([action])

class GPAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)
    def choose_action(self, state, h):
        state = utils.obs2bin(state)
        policy_raw = self.policy_all[state[0],state[1],state[2], :, h]
        policy = policy_raw/policy_raw.sum() 
        action = np.random.choice(np.linspace(-2, 2, utils.num_act_bins), p=policy)
        return np.array([action])

class FQIAgent(): 
    def __init__(self, policy_log):
        self.policy_all = np.load(policy_log)
    def choose_action(self, state, h):
        state = utils.obs2bin(state)
        policy_raw = self.policy_all[state[0],state[1],state[2], :]
        policy = policy_raw/policy_raw.sum() 
        action = np.random.choice(np.linspace(-2, 2, utils.num_act_bins), p=policy)
        return np.array([action])

class PQL_BCQ_Agent(): 
    def __init__(self, policy_log): 
        with open(policy_log, 'rb') as inp:
            self.policy = pickle.load(inp)
    def choose_action(self, state, h):
        return self.policy.select_action(np.array(state))
