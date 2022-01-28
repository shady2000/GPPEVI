import numpy as np 
import torch 
from env import ACTIONS

num_trajectories = 500
state_dim = 2
action_dim = 1

def phi(state, action): 
    x = state[0]
    y = state[1]
    action_index = ACTIONS.index(action)
    return np.array([x, y, action_index])

class Dataset(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0 
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1)) 
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr+1)%self.max_size
        self.size = min(self.size+1, self.max_size)
        
        
    def sample(self, batch_size, use_bootstrap=False):
        ind = np.random.randint(0, self.size, size=batch_size)
        if use_bootstrap:
            return (
			torch.FloatTensor(self.state[ind]),
			torch.FloatTensor(self.action[ind]),
			torch.FloatTensor(self.next_state[ind]),
			torch.FloatTensor(self.reward[ind]),
			torch.FloatTensor(self.not_done[ind]),
			torch.FloatTensor(self.bootstrap_mask[ind]),
		)
        return (
			torch.FloatTensor(self.state[ind]),
			torch.FloatTensor(self.action[ind]),
			torch.FloatTensor(self.next_state[ind]),
			torch.FloatTensor(self.reward[ind]),
			torch.FloatTensor(self.not_done[ind])
		)
    
    def save(self, save_folder):
        np.save(f"{save_folder}/state.npy", self.state[:self.size])
        np.save(f"{save_folder}/action.npy", self.action[:self.size])
        np.save(f"{save_folder}/next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}/reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}/not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}/ptr.npy", self.ptr)

    def load(self, save_folder, size=-1, bootstrap_dim=None):
        reward_buffer = np.load(f"{save_folder}/reward.npy")
		
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy")[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy")[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/next_state.npy")[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}/not_done.npy")[:self.size]
        if bootstrap_dim is not None:
            self.bootstrap_dim = bootstrap_dim
            bootstrap_mask = np.random.binomial(n=1, size=(1, self.size, bootstrap_dim,), p=0.8)
            bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
            self.bootstrap_mask = bootstrap_mask[:self.size]