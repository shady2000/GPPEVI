from tkinter import Grid
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np
import gym
import utils
import random

class FNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.linin = nn.Linear(state_size+action_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.linout = nn.Linear(128, 1)
        self.rl = nn.ReLU(inplace=True)

        self.linout.bias.data.fill_(1e-3)
        self.linout.weight.data.fill_(1e-3)

    def forward(self, z):
        if not torch.is_tensor(z):
            z = np.array([z[0], z[1], z[2]])
            z = torch.from_numpy(z)
        z = z.to(torch.float)
        z = self.linin(z)
        z = self.rl(z)
        z = self.fc1(z)
        z = self.rl(z)
        z = self.linout(z)
        z = self.rl(z) 
        z = z.to(torch.float64)
        return z

#load dataset and init uniform policy
def init():
    replay_buffer = utils.Dataset(utils.state_dim, utils.action_dim)
    replay_buffer.load("./pgtab_dataset--nt-{}_h-{}_s-{}_a-{}".format(utils.num_trajectories, utils.HORIZON, utils.num_obs_bins, utils.num_act_bins))
    policy = np.full([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim, 1/utils.num_act_bins)
    states, actions, rewards, next_states = replay_buffer.state, replay_buffer.action, replay_buffer.reward, replay_buffer.next_state
    states, actions, next_states = utils.obs2bin(states), utils.act2bin(actions), utils.obs2bin(next_states)
    return states, actions, rewards, next_states, policy

def main(args):

    #INPUT: pi: numpy array of shape (BOARD_SIZE, BOARD_SIZE, len(ACTIONS), s is a state, f is a network 
    #OUTPUT: f(s, pi) = E_{s' \sim pi(.|a)}[f(s,a)], type=float
    def f_s_pi(f, s, pi):
        pi_s = pi[int(s[0]), int(s[1]), int(s[2]), :]
        pi_s = torch.from_numpy(pi_s).to(torch.float64)
        arr_f = [f(np.array([s[0], s[1], s[2], a])) for a in range(utils.num_act_bins)]
        arr_f = torch.Tensor(arr_f).to(torch.float64)
        return torch.dot(arr_f, pi_s)

    #INPUT: f1, f: networks f' and f respectively; a policy pi, states, actions, rewards, next_states
    #OUTPUT: L as defined in the paper
    def L(f1, f, pi, states, actions, rewards, next_states):
        z = np.concatenate((states, actions), axis=1)
        arr_f1 = f1(z)
        arr_f1 = arr_f1.squeeze(-1)
        
        r = rewards.squeeze(-1)
        arr_f = [f_s_pi(f, s, pi) for s in next_states]
        arr_f = torch.Tensor(arr_f).to(torch.float64)
        
        ret_val = ((arr_f - r - arr_f)**2).mean()
        
        return ret_val 

    #INPUT: f: a network, dataset
    #OUTPUT: min_{f' \in F}L(f', f, pi, D)
    def min_L(f, pi, states, actions, rewards, next_states):
        X_train = np.concatenate((states, actions), axis=1) #define X_train
        X_train = torch.from_numpy(X_train).to(torch.float64)
        
        f_temp = FNetwork(utils.state_dim, utils.action_dim) #define network

        r = rewards.squeeze(-1)
        r = torch.Tensor(r)
        arr_f = [f_s_pi(f, s, pi) for s in next_states]
        arr_f = torch.Tensor(arr_f).to(torch.float64)
        Y_train = r + arr_f #define Y_train
        
        loss_fn = torch.nn.MSELoss(reduction='sum') #define loss function
        optimizer = RMSprop(f_temp.parameters(), lr=args.lr_L) #define optimizer

        #Start training to minimize L(f', f, pi, D)
        for t in range(args.iter_min_L):
            Y_pred = f_temp(X_train)
            loss = loss_fn(Y_pred, Y_train)
            #print(Y_pred[0:10])
            if t%10 == 9:
                print(t, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return L(f_temp, f, pi, states, actions, rewards, next_states)

    #INPUT: f, policy, dataset
    #OUTPUT: Var_Epsilon as defined at formula 3.1 in the paper
    def var_epsilon(f, pi, states, actions, rewards, next_states):
        return L(f, f, pi, states, actions, rewards, next_states) - min_L(f, pi, states, actions, rewards, next_states)

    #initialize things
    states, actions, rewards, next_states, policy = init()
    f_curr = FNetwork(utils.state_dim, utils.action_dim) #init f0
    s0 = np.array([utils.num_obs_bins-1, utils.num_obs_bins-1]) #init state
    f_optimizer = Adam(f_curr.parameters(), lr=args.lr_func) #optimizer for f_curr
    sum_policy = np.zeros((policy.shape)) #to store the final uniform policy

    #update f_t and policy
    for t in range(args.T):
        print("Updating function and policy at iteration", t)
        # minimizing to have f_{t+1}
        for it in range(args.iter_train_F):
            rand_id = random.sample(range(utils.num_trajectories), args.batch_size)
            s, a, r, s_next = states[rand_id], actions[rand_id], rewards[rand_id], next_states[rand_id] #sample a batch from dataset 
            loss_f = f_s_pi(f_curr, s0, policy) + args.lamda*var_epsilon(f_curr, policy, s, a, r, s_next)
            loss_f.requires_grad=True #attent to this thing
            f_optimizer.zero_grad()
            loss_f.backward()
            f_optimizer.step() #update func parameters 

            if it%20 == 19:
                print(loss_f)
                
        #updating policy
        for s_id in np.ndindex((utils.num_obs_bins,)*utils.state_dim): 
            for a_id in np.ndindex((utils.num_act_bins,)*utils.action_dim): 
                s = np.asarray(s_id)
                a = np.asarray(a_id)
                z = np.array([s[0], s[1], s[2], a]).astype(float)
                policy[s_id][a_id] = policy[s_id][a_id]*np.exp(args.mu*f_curr(z).detach().numpy())
            policy[s][:] =  policy[s][:]/np.sum(policy[s][:])
        sum_policy += policy

    #return uniformly mixed policy
    uniformly_mixed_policy = sum_policy/args.T
    return uniformly_mixed_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=float, default=1)   #discount factor
    parser.add_argument('--lamda', type=float, default=0.1) #regularizing hyper param 
    parser.add_argument('--batch_size', default=200)    #batch size 
    parser.add_argument('--T', type=int, default=500)   #number of iterations to update f_t and policy
    parser.add_argument('--mu', type=float, default=1)  #hyper parameter at the step updating policy
    parser.add_argument('--iter_train_F', type=int, default=200)    #number of iterations to find f_{t+1}
    parser.add_argument('--lr_func', type= float, default=1e-3) #learning rate for updating f_t
    parser.add_argument('--iter_min_L', type=int, default=500)  #number of iterations to find min_f' L(f', f, pi, D)
    parser.add_argument('--lr_L', type=float, default=1e-3) #learning rate to find min_f' L(f', f, pi, D)

    args = parser.parse_args()
    policy = main(args)

    np.save("bellman_nt-{}".format(utils.num_trajectories), policy)
    print("Saved Bellman Consistent policy to", "bellman_nt-{}.npy".format(utils.num_trajectories))