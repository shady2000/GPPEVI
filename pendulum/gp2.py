import math
import gym
import numpy as np
import utils
from utils import HORIZON, env_name, phi
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
    #return np.array([kp(z, i, zi, j) for (zi, j) in tilda_col])
    return retval

def z(s, a):
    return np.concatenate(s, a)

def init():
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    dataset.load("./dataset--nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    env = gym.make(env_name)
    K = utils.num_trajectories
    d = phi(env.reset(), env.action_space.sample()).shape[0]
    prob = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[HORIZON])
    VHat = np.zeros([utils.num_obs_bins]*utils.state_dim+[HORIZON+1]) 
    states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
    rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))
    next_states = dataset.next_state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    S_next = np.ravel(next_states, order='F')
    m = len(env.reset())

    Z_all = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[utils.state_dim+utils.action_dim])
    Z_tilda_all = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[m]+[utils.state_dim+utils.action_dim])

    for s_id in np.ndindex((utils.num_obs_bins,)*utils.state_dim): 
        for a_id in np.ndindex((utils.num_act_bins,)*utils.action_dim): 
            s = np.asarray(s_id)
            a = np.asarray(a_id)
            z = np.concatenate((s, a))
            Z_all[s_id][a_id] = Z_all[s_id][a_id][:] = z
            for j in range(m): 
                Z_tilda_all[s_id][a_id][j][:] = np.concatenate((z, np.asarray([j]))) 

    Z_all = Z_all.reshape(-1, utils.state_dim+utils.action_dim)
    Z_tilda_all = Z_tilda_all.reshape(-1,utils.state_dim+utils.action_dim+1)

    return dataset, env, prob, VHat, states, S_next, actions, rewards, K, d, m, Z_all, Z_tilda_all

def main(args):
    _, env, prob, VHat, states, S_next, actions, rewards, K, d, m, Z_all, Z_tilda_all = init()
    z = np.concatenate((states, actions), axis=1)
    z_tilda = np.zeros((K*m, utils.state_dim+utils.action_dim+1))
    for id in range(K*m*(utils.state_dim+utils.action_dim+1)):
        i = id//(K*m)
        j = id%(K*m)
        pair = np.concatenate((states[j], actions[j]))
        z_tilda[id] = np.zeros((K*m, utils.state_dim+utils.action_dim+1))
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) 

    gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_r.fit(z, rewards)
    mu_r, sigma_r = gp_r.predict(Z_all, return_std=True)

    gp_p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_p.fit(z_tilda, S_next)
    mu_p, sigma_p = gp_p.predict(Z_tilda_all, return_std=True)

    K_r = np.zeros((K*HORIZON, K*HORIZON))
    K_p = np.zeros((K*m, K*m))
    for i in range(K*HORIZON):
        for j in range(K*HORIZON): 
            K_r[i][j] = kr(z[i], z[j])
    for u in range(K*HORIZON*m):
        for v in range(K*HORIZON*m):
            K_p[u][v] = kp(z_tilda[u][0:-1], z_tilda[u][-1], z_tilda[v][0:-1], z_tilda[v][-1])

    mig_r = 1/2*math.log(np.linalg.det(np.identity(K*HORIZON)+ 1/args.lr*K_r))
    mig_p = 1/2*math.log(np.linalg.det(np.identity(K*m*HORIZON) + 1/args.lp*K_p))
    """ d = z[0].shape()
    mig_rh = math.log(K)**d
    mig_ph = math.log(K*m)**d """

    beta_r = 1 + args.sdR/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_r)
    beta_p = 1 + args.sdP/math.sqrt(m*HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_p)

    big_gamma_h = bv = Q_overline_h = Q_hat_h = pi_h = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)

    mu_r = mu_r.reshape(Z_all.shape)
    sigma_r = sigma_r.reshape(Z_all.shape)
    mu_p = mu_p.reshape(Z_tilda_all.shape)
    sigma_p = sigma_p.reshape(Z_tilda_all.shape)
    for h in list(range(1, HORIZON+1))[::-1]: 
        print("Running at period h = ", h)
        for x in np.ndindex((utils.num_obs_bins,)*utils.state_dim):
            for a in np.ndindex((utils.num_act_bins,)*utils.action_dim): 
                big_gamma_h[x][a] = beta_r*sigma_r[x][a] + HORIZON*beta_p*np.dot(sigma_p[x][a], sigma_p[x][a])
                bv[x][a] = mu_r[x][a] + np.sum(VHat[:, :, h]*np.dot(mu_p[:,:,a], mu_p[:,:,a]))
                big_gamma_h = np.maximum(0.01, big_gamma_h)
                Q_overline_h[x][a] = np.random.normal(loc = bv[x][a], scale = big_gamma_h[x][a])
                Q_hat_h = np.clip(0, HORIZON-h+1, Q_overline_h)
            for a in np.ndindex((utils.num_act_bins,)*utils.action_dim):
                if (a == np.argmax(Q_hat_h[x][:])):
                    #pi_h[x][a] = 1 - args.exp
                    pi_h[x][a] = 1
                else:
                    #pi_h[x][a] = args.exp/(utils.num_act_bins)
                    pi_h[x][a] = 0
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