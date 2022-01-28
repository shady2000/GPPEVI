import math
import numpy as np
import utils
from env import BOARD_SIZE, HORIZON, ACTIONS
from env import GridWorld
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def kr(z1, z2): 
    #squared exponential kernel
    #z = np.array([s[0], s[1], a]) 
    #return np.exp(-np.dot(z1-z2, z1-z2)/args.sigma)
    return np.dot(z1-z2, z1-z2)

def kp(z1, i1, z2, i2):
    z_1 = np.concatenate((z1, np.array([i1])))
    z_2 = np.concatenate((z2, np.array([i2])))
    return np.exp(-np.dot(z_1-z_2, z_1-z_2)/args.sigma)

def kr_col(z, col):
    return np.array([kr(z, zi) for zi in col])

def kp_col(z, i, tilda_col):
    retval = np.zeros((len(tilda_col)))
    for count, elem in enumerate(tilda_col): 
        s0, s1, a, j = elem
        zi = np.array([s0, s1, a])
        retval[count] = kp(z, i, zi, j)
    return retval

def z(s, a):
    return np.concatenate(s, a)

def create_cov_matrix(k_func, data):
    return k_func(data)

def init():
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    dataset.load("./dataset--nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    env = GridWorld()
    K = utils.num_trajectories
    d = utils.phi(env.current_location, ACTIONS[0]).shape[0]
    prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]+[HORIZON])
    VHat = np.zeros([BOARD_SIZE]*utils.state_dim+[HORIZON+1]) 
    states = dataset.state[0:K*HORIZON,:]
    actions = dataset.action[0:K*HORIZON,:]
    rewards = dataset.reward[0:K*HORIZON]
    next_states = dataset.next_state[0:K*HORIZON,:]
    S_next = np.ravel(next_states, order='F')
    Z_all = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), 3))
    m = len(env.current_location)
    Z_tilda_all = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m, 4))
    for x in np.ndindex((BOARD_SIZE, BOARD_SIZE)):
        for a in range(len(ACTIONS)):
            Z_all[x][a][:] = np.array([x[0], x[1], ACTIONS[a]])
    for x in np.ndindex((BOARD_SIZE, BOARD_SIZE)):
        for a in range(len(ACTIONS)): 
            for i in range(m): 
                Z_tilda_all[x][a][i][:] = np.array([x[0], x[1], ACTIONS[a], i])
    Z_all = Z_all.reshape(-1, 3)
    Z_tilda_all = Z_tilda_all.reshape(-1,4)

    return dataset, env, prob, VHat, states, S_next, actions, rewards, K, d, m, Z_all, Z_tilda_all

def main(args):
    _, env, prob, VHat, states, S_next, actions, rewards, K, d, m, Z_all, Z_tilda_all = init()
    z = np.concatenate((states, actions), axis=1)
    z_tilda = np.zeros((K*HORIZON*m, 4)) 

    for id in range(K*HORIZON*m):
        if id<=K*HORIZON-1:
            z_tilda[id] = np.array([states[id][0], states[id][1], actions[id], 1], dtype=object)
        else: 
            z_tilda[id] = np.array([states[id-K*HORIZON][0], states[id-K*HORIZON][1], actions[id-K*HORIZON], 2], dtype=object)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) 

    gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_r.fit(z, rewards) # z la X, rewards la y
    print("Done fitting R")
    mu_r, sigma_r = gp_r.predict(Z_all, return_std=True)
    print("Done predicting R")
    gp_p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_p.fit(z_tilda, S_next)
    mu_p, sigma_p = gp_p.predict(Z_tilda_all, return_std=True)
    print("Done for p")
    
    K_r = np.zeros((K*HORIZON, K*HORIZON))
    K_p = np.zeros((K*m, K*m))
    for i in range(K*HORIZON):
        for j in range(K*HORIZON): 
            K_r[i][j] = kr(z[i], z[j])
    for u in range(K*HORIZON*m):
        for v in range(K*HORIZON*m):
            K_p[u][v] = kp(z_tilda[u][0:-1], z_tilda[u][-1], z_tilda[v][0:-1], z_tilda[v][-1])

    mig_r = 1/2*math.log(np.linalg.det(np.identity(K*HORIZON)+ 1/args.lr*K_r))
    mig_p = 1/2*math.log(np.linalg.det(np.identity(K*HORIZON*m) + 1/args.lp*K_p))

    """ d = z[0].shape()
    mig_rh = math.log(K)**d
    mig_ph = math.log(K*m)**d """

    beta_r = 1 + args.sdR/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_r)
    beta_p = 1 + args.sdP/math.sqrt(m*HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_p)

    #beta_r = beta_p = 1

    big_gamma_h = bv = Q_overline_h = Q_hat_h = pi_h = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))

    mu_r = mu_r.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
    sigma_r = sigma_r.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
    mu_p = mu_p.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m))
    sigma_p = sigma_p.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m))

    for h in list(range(1, HORIZON+1))[::-1]: 
        print("Running at period h = ", h)
        for x in np.ndindex((BOARD_SIZE,)*utils.state_dim):
            for a in range(len(ACTIONS)): 
                big_gamma_h[x][a] = beta_r*sigma_r[x][a] + HORIZON*beta_p*np.sqrt(sigma_p[x][a][0]**2 + sigma_p[x][a][1]**2)
                bv[x][a] = mu_r[x][a] + np.sum(VHat[:, :, h]*np.sqrt(mu_p[:,:,a,0]**2 + mu_p[:,:,a,1]**2))
                big_gamma_h = np.maximum(0.01, big_gamma_h)
                Q_overline_h[x][a] = np.random.normal(loc = bv[x][a], scale = big_gamma_h[x][a])
                Q_hat_h = np.clip(0, HORIZON-h+1, Q_overline_h)
            for a in range(len(ACTIONS)):
                if (a == np.argmax(Q_hat_h[x][:])):
                    pi_h[x][a] = 1 - args.exp
                else:
                    pi_h[x][a] = args.exp/(len(ACTIONS))
            VHat[x][h-1] = np.dot(Q_hat_h[x][:], pi_h[x][:])
        prob[:,:,:,h-1] = pi_h
    return prob

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1)
    parser.add_argument('--lp', default=1)
    parser.add_argument('--exp', default=0.1)
    parser.add_argument('--sdR', default=1)
    parser.add_argument('--sdP', default=1)
    parser.add_argument('--delta', default=0.1)
    parser.add_argument('--sigma', default=1)
    args = parser.parse_args()
    prob = main(args)
    np.save("gp_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved offline GP policy to", "gp_nt-{}_h-{}.npy".format(utils.num_trajectories, HORIZON))