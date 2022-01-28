from os import stat
import utils 
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
import random
import time
from env import ACTIONS, BOARD_SIZE, GridWorld, HORIZON

def init(): 
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    dataset.load("./dataset--nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    env = GridWorld()
    Z_all = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), 3))
    for s in np.ndindex((BOARD_SIZE,)*utils.state_dim):
        for a in range(len(ACTIONS)): 
            Z_all[s][a][:] = np.array([s[0], s[1], ACTIONS[a]])
    Z_all = Z_all.reshape((-1, utils.state_dim+utils.action_dim))
    states, actions, rewards, next_states = dataset.state, dataset.action, dataset.reward, dataset.next_state
    z = np.concatenate((states, actions), axis=1)

    prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]+[HORIZON])

    return env, dataset, Z_all, states, actions, rewards, next_states, z, prob

def main(args):
    env, dataset, Z_all, states, actions, rewards, next_states, z_data, prob = init()
    kernel = ConstantKernel(1.0,constant_value_bounds=(1e-2, 10)) * RBF(length_scale=1.0,length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,1))
    gp_Q = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    Y = np.zeros((len(states), 1))
    Q_pred = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)])
    iter = 0 
    print("== Start Training ==")

    while (iter < args.it):
        print("Training at iteration", iter)
        for i in range(len(states)):
            Y[i] = rewards[i] + args.gamma*np.amax(Q_pred[int(next_states[i][0]), int(next_states[i][1])])
        random_idx = random.sample(range(len(states)), args.N)
        gp_Q = gp_Q.fit(z_data[random_idx], Y[random_idx])
        Q_pred = gp_Q.predict(Z_all)
        Q_pred = Q_pred.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        iter = iter+1

    print("== Finished Training ==")

    prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)])
    Q_pred = Q_pred.reshape([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]*utils.action_dim)
    for x in np.ndindex((BOARD_SIZE,)*utils.state_dim):
        for a in range(len(ACTIONS)):
            if (a == np.argmax(Q_pred[x][:])):
                prob[x][a] = 1 - args.exp
            else:
                prob[x][a] = args.exp/(len(ACTIONS)-1)

    return prob 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--it', type=int, default=100)
    parser.add_argument('--N', type = int, default=200)
    parser.add_argument('--exp', type = float, default=0.1)
    args = parser.parse_args()
    prob = main(args)
    np.save("gpfqi_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved offline policy to", "gpfqi_nt-{}_h-{}.npy".format(utils.num_trajectories, HORIZON))