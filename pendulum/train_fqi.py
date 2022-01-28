from os import stat
import gym 
import utils 
from utils import obs2bin, HORIZON
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
import random
import time


#implements this paper: http://web.mit.edu/miaoliu/www/publications/GPQ_AAS_Special_issue.pdf

def init(): 
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    dataset.load("./dataset--nt-{}_h-{}_s-{}_a-{}".format(utils.num_trajectories, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    env = gym.make(utils.env_name)
    Z_all = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[utils.state_dim+utils.action_dim])
    for s_id in np.ndindex((utils.num_obs_bins,)*utils.state_dim):
        for a_id in np.ndindex((utils.num_act_bins)*utils.action_dim): 
            s = np.asarray(s_id) 
            a = np.asarray(a_id)
            Z_all[s_id][a_id][:] = np.concatenate((s, a))
    Z_all = Z_all.reshape((-1, utils.state_dim+utils.action_dim))
    #Z_all stores all (s, a) pairs
    states, actions, rewards, next_states = dataset.state, dataset.action, dataset.reward, dataset.next_state
    z = np.concatenate((states, actions), axis=1)
    #z stores all (s,a) pairs from the dataset 
    s_init = obs2bin(env.reset())
    prob = np.zeros([utils.num_obs_bins]*len(s_init)+[utils.num_act_bins]+[HORIZON])

    return env, dataset, Z_all, states, actions, rewards, next_states, z, prob

def main(args):
    env, dataset, Z_all, states, actions, rewards, next_states, z_data, prob = init()
    kernel = ConstantKernel(1.0,constant_value_bounds=(1e-2, 10)) * RBF(length_scale=1.0,length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,1))
    gp_Q = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    Y = np.zeros((len(states), 1))
    Q_pred = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins])
    iter = 0 
    print("== Start Training ==")

    while (iter < args.it):
        print("Training at iteration", iter)
        for i in range(len(states)):
            Y[i] = rewards[i] + args.gamma*np.amax(Q_pred[utils.obs2bin(next_states[i])])
        random_idx = random.sample(range(len(states)), args.N)
        print(z_data.shape)
        print(Y.shape)
        gp_Q = gp_Q.fit(z_data[random_idx], Y[random_idx])
        Q_pred = gp_Q.predict(Z_all)
        iter = iter+1
    print("== Finished Training ==")

    prob = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins])
    Q_pred = Q_pred.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
    for x in np.ndindex((utils.num_obs_bins,)*utils.state_dim):
        for a in range(utils.num_act_bins):
            if (a == np.argmax(Q_pred[x][:])):
                prob[x][a] = 1 #- args.exp
            else:
                prob[x][a] = 0 #args.exp/(utils.num_act_bins-1)

    return prob 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--it', type=int, default=1000)
    parser.add_argument('--N', type = int, default=200)
    parser.add_argument('--exp', type = float, default=0.1)
    args = parser.parse_args()
    prob = main(args)
    np.save("gpfqi_nt-{}_s-{}_a-{}".format(utils.num_trajectories, utils.num_obs_bins, utils.num_act_bins), prob)
    print("Saved offline policy to", "gpfqi_nt-{}_s-{}_a-{}.npy".format(utils.num_trajectories, HORIZON, utils.num_obs_bins, utils.num_act_bins))