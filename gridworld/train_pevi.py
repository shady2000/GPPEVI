import math
from env import BOARD_SIZE, HORIZON, ACTIONS, phi
import numpy as np
from env import GridWorld
import utils 

ps = []

def init(): 
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    dataset.load("./dataset--nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    env = GridWorld()
    K = utils.num_trajectories
    d = phi(env.current_location, ACTIONS[0]).shape[0]
    prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]+[HORIZON])
    VHat = np.zeros([BOARD_SIZE]*utils.state_dim+[HORIZON+1]) 
    states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
    rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))
    return dataset, env, prob, VHat, states, actions, rewards, K, d

def main(args):
    dataset, env, prob, VHat, states, actions, rewards, K, d = init()
    #Hyperparameters 
    exp = args.exp
    lamb = 1
    shrink_rate = 0.5
    beta_init = 2000
    #epsilon = args.epsilon
    #xi = math.log(2*d*HORIZON*K/epsilon)
    #lamb = 1
    #c = 20
    #beta = c*d*HORIZON*math.sqrt(xi)
    K = utils.num_trajectories 
    VHat = np.zeros([BOARD_SIZE, BOARD_SIZE, HORIZON+1]) 
    state = env.current_location
    d = phi(state, ACTIONS[0]).shape[0]

    for h in list(range(1, HORIZON+1))[::-1]:
        print("Running at period", h)
        states_h, actions_h, rewards_h = states[:, h-1, :], actions[:, h-1, :], rewards[:, h-1]
        big_lambd = lamb*np.identity(d)
        var = 0
        for tau in range(K): 
            s_tau_h, a_tau_h, r_tau_h = states_h[tau], actions_h[tau], rewards_h[tau]
            big_lambd = big_lambd + np.matmul(phi(s_tau_h, a_tau_h), phi(s_tau_h, a_tau_h).T)
            id = tuple(s_tau_h.astype(np.int32).reshape(1, -1)[0])
            var = var + phi(s_tau_h, a_tau_h)*(r_tau_h + VHat[id][h])
        w_hat_h = np.linalg.inv(big_lambd).dot(var)

        Big_Gamma_h = Q_Overline_h = Q_hat_h = pi_h = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)])
        for x in np.ndindex((BOARD_SIZE,)*utils.state_dim):
            for a in range(len(ACTIONS)):
                beta = beta_init
                s = np.asarray(x)
                Big_Gamma_h[x][a] = math.sqrt(np.matmul(np.matmul(phi(s,a).T, np.linalg.inv(big_lambd)),phi(s,a)))
                while (np.dot(phi(s,a).T, w_hat_h) - beta*Big_Gamma_h[x][a] < 0 and beta > 0.01):
                    beta = beta*shrink_rate
                Q_Overline_h[x][a] = np.dot(phi(s,a).T, w_hat_h) - beta*Big_Gamma_h[x][a]
                Q_hat_h[x][a] = np.clip(Q_Overline_h[x][a], 0, HORIZON - h + 1)
            for a in range(len(ACTIONS)):
                if (a == np.argmax(Q_hat_h[x][:])):
                    pi_h[x][a] = 1 - args.exp
                else:
                    pi_h[x][a] = args.exp/(len(ACTIONS))
            VHat[x][h-1] = np.dot(Q_hat_h[x][:], pi_h[x][:])
        prob[:, :, :, h-1] = pi_h
    return prob

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--exp', type=float, default=0.1)
    #parser.add_argument('--lamb', type=float, default=1)
    #parser.add_argument('--beta', type=float)
    args = parser.parse_args()
    prob = main(args)
    np.save("pevi_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved offline policy to", "pevi_nt-{}_h-{}.npy".format(utils.num_trajectories, HORIZON))