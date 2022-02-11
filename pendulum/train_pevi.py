import math
import gym
from utils import obs2bin
from utils import HORIZON, phi
import numpy as np
import utils 

data_path = "/media/Z/shun/storage/pendulum/dataset"
model_path = "/media/Z/shun/storage/pendulum/model/pevi"

num_t = utils.num_trajectories
num_s = utils.num_obs_bins
num_a = utils.num_act_bins

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--exp', type=float, default=0.1)
#parser.add_argument('--lamb', type=float, default=1)
#parser.add_argument('--beta', type=float, default=1)
args = parser.parse_args()

model_dir = f"{model_path}/pevi_phitrue_ddpg_sample_nt-{num_t}_h-{HORIZON}_s-{num_s}_a-{num_a}"

dataset = utils.Dataset(utils.state_dim, utils.action_dim)
dataset.load(data_path + f"/ddpg_dataset--nt-{num_t}_h-{HORIZON}")
env = gym.make(utils.env_name)
env.seed(1)
s_init = obs2bin(env.reset())
K = utils.num_trajectories
d = phi(env.reset(), env.action_space.sample()).shape[0]
prob = np.zeros([utils.num_obs_bins]*len(s_init)+[utils.num_act_bins]+[HORIZON])
VHat = np.zeros([utils.num_obs_bins]*len(s_init)+[HORIZON+1]) 
states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))
#states = utils.obs2bin(states)
#actions = utils.obs2bin(actions)

# Hyperparameters
epsilon = args.epsilon
lamb = 1
c = 1

beta_init = 2000
shrink_rate = 0.5

xi = math.log(2*d*HORIZON*K/epsilon)
# beta = c*d*HORIZON*math.sqrt(xi)

def train():
    for h in list(range(1, HORIZON+1))[::-1]: 
        print("Running at period", h)
        beta = beta_init
        states_h = states[:, h-1, :]
        actions_h = actions[:, h-1, :]
        rewards_h = rewards[:, h-1]
        big_lambd = lamb*np.identity(d)
        var = 0
        for tau in range(K): 
            s_tau_h = states_h[tau]
            a_tau_h = actions_h[tau]
            r_tau_h = rewards_h[tau]
            big_lambd = big_lambd + np.matmul(phi(s_tau_h, a_tau_h), phi(s_tau_h, a_tau_h).T)
            var = var + phi(s_tau_h, a_tau_h)*(r_tau_h + VHat[int(s_tau_h[0]),int(s_tau_h[1]),int(s_tau_h[2]), h])
        w_hat_h = np.linalg.inv(big_lambd).dot(var) 
        
        Big_Gamma_h = Q_Overline_h = Q_hat_h = pi_h = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins])
        
        for x in np.ndindex((utils.num_obs_bins,)*utils.state_dim):
            for a_id in range(utils.num_act_bins):
                s = np.asarray(x)
                a = np.array([a_id])
                Big_Gamma_h[x][a_id] = math.sqrt(np.matmul(np.matmul(phi(s,a).T, np.linalg.inv(big_lambd)),phi(s,a)))
                while (np.dot(phi(s,a).T, w_hat_h) - beta*Big_Gamma_h[x][a] < 0 and beta > 0.01):
                    beta = beta*shrink_rate
                #Q_Overline_h[x][a] = np.dot(phi(s,a).T, w_hat_h) - beta*Big_Gamma_h[x][a]
                Q_Overline_h[x][a] = np.random.normal(loc = np.dot(phi(s,a).T, w_hat_h), scale = beta*Big_Gamma_h[x][a])
                Q_hat_h[x][a] = np.clip(Q_Overline_h[x][a], 0, HORIZON - h + 1)
            for a in range(utils.num_act_bins):
                if (a == np.argmax(Q_hat_h[x][:])):
                    #pi_h[x][a] = 1 - args.exp
                    pi_h[x][a] = 1 - args.exp
                else:
                    #pi_h[x][a] = args.exp/(utils.num_act_bins-1)
                    pi_h[x][a] = 0
            VHat[x][h-1] = np.dot(Q_hat_h[x][:], pi_h[x][:])
            prob[:, :, :, :, h-1] = pi_h

prob = train()
np.save(model_dir, prob)
print(f"Saved offline policy to {model_dir}.npy")


    
