import math
import gym
import numpy as np
import utils
from utils import HORIZON, env_name, phi
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
import random
import time

#specifying the kernel functions 
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
    dataset.load("./pg_dataset--nt-{}_h-{}_s-{}_a-{}".format(utils.num_trajectories, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    env = gym.make(env_name)
    env.seed(1)
    K = utils.num_trajectories
    d = phi(env.reset(), env.action_space.sample()).shape[0]
    prob = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[HORIZON])
    VHat = np.zeros([utils.num_obs_bins]*utils.state_dim+[HORIZON+1]) 
    states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
    rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))
    next_states = dataset.next_state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    m = len(env.reset())

    Z_all = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[utils.state_dim+utils.action_dim])
    #Z_tilda_all = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim+[m]+[utils.state_dim+utils.action_dim])
    #Z_tilda_all is for the setting of Online Learning paper
    for s_id in np.ndindex((utils.num_obs_bins,)*utils.state_dim): 
        for a_id in np.ndindex((utils.num_act_bins,)*utils.action_dim): 
            s = np.asarray(s_id)
            a = np.asarray(a_id)
            z = np.concatenate((s, a))
            Z_all[s_id][a_id] = Z_all[s_id][a_id][:] = z
            #for j in range(m): 
            #    Z_tilda_all[s_id][a_id][j][:] = np.concatenate((z, np.asarray([j]))) 

    Z_all = Z_all.reshape(-1, utils.state_dim+utils.action_dim)
    #Z_tilda_all = Z_tilda_all.reshape(-1,utils.state_dim+utils.action_dim+1)

    states = utils.obs2bin(states)
    actions = utils.obs2bin(actions)
    next_states = utils.obs2bin(next_states)

    return dataset, env, prob, VHat, states, next_states, actions, rewards, K, d, m, Z_all #, Z_tilda_all

def main(args):
    _, env, prob, VHat, states, next_states, actions, rewards, K, d, m, Z_all = init()
    kernel = ConstantKernel(1.0,constant_value_bounds=(1e-2, 10)) * RBF(length_scale=1.0,length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,1))
    gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_px = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_py = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_ps = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    for h in list(range(1, HORIZON+1))[::-1]: 
        print("Running at period h = ", h)
        states_h, actions_h, rewards_h, next_states_h = states[:, h-1, :], actions[:, h-1, :], rewards[:, h-1], next_states[:, h-1, :]
        states_xh = states_h[:, 0].reshape(-1, 1)
        states_yh = states_h[:, 1].reshape(-1, 1)
        states_sh = states_h[:, 2].reshape(-1, 1)
        z_h = np.concatenate((states_h, actions_h), axis=1)
        z_xh = np.concatenate((states_xh, actions_h), axis=1)
        z_yh = np.concatenate((states_yh, actions_h), axis=1)
        z_sh = np.concatenate((states_sh, actions_h), axis=1)

        """ z_tilda_h is for the setting of Online Learning paper 
        z_tilda_h = np.zeros((K*m, utils.state_dim+utils.action_dim+1))
        for id in range(K*m):
            i = id//m
            j = id%m
            z = np.concatenate((states_h[j], actions_h[j]))
            z_tilda_h[id] = np.concatenate((z, np.asarray([i])))
        S_h = np.ravel(next_states_h, order='F')
 """
        #Initiating and fitting Gaussian Processes 
        random_idx = random.sample(range(utils.num_trajectories), args.batch_size)
        start_time = time.time()

        gp_r.fit(z_h[random_idx], rewards_h[random_idx])
        mu_rh, sigma_rh = gp_r.predict(Z_all, return_std=True)

        gp_px.fit(z_xh[random_idx], next_states_h[random_idx, 0])
        mu_phx, sigma_phx = gp_px.predict(Z_all[:, [0,3]], return_std=True)

        gp_py.fit(z_yh[random_idx], next_states_h[random_idx, 1])
        mu_phy, sigma_phy = gp_py.predict(Z_all[:, [1,3]], return_std=True)

        gp_ps.fit(z_sh[random_idx], next_states_h[random_idx, 2])
        mu_phs, sigma_phs = gp_ps.predict(Z_all[:, [2,3]], return_std=True)

        mu_rh = mu_rh.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        sigma_rh = sigma_rh.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        mu_phx = mu_phx.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        sigma_phx = sigma_phx.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        mu_phy = mu_phy.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        sigma_phy = sigma_phy.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        mu_phs = mu_phs.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        sigma_phs = sigma_phs.reshape([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)
        
        print("Time taken for all GP update is", time.time()-start_time)
        start_time = time.time()

        # Computing W matrix 
        tau = args.lamda*np.eye(3)
        cov = np.array([[args.var, 0, 0], [0, args.var, 0], [0, 0, args.var]])
        #mean = np.array([mu_phx, mu_phy, mu_phs])
        M = np.linalg.inv(tau+cov)
        alpha = np.sqrt(np.linalg.det(M)*args.lamda)*args.sigma
        Wh = np.zeros((utils.num_obs_bins**utils.state_dim, utils.num_obs_bins**utils.state_dim))
        for i in range(utils.num_obs_bins**utils.state_dim):
            for j in range(utils.num_obs_bins**utils.state_dim):
                xy = i%utils.num_obs_bins**2
                s = i//utils.num_obs_bins**2
                x = xy%utils.num_obs_bins
                y = xy//utils.num_obs_bins
                a_temp = np.argmax(prob[x,y,s,:,h-1])
                mu_temp_x = mu_phx[x, y, s, a_temp] 
                mu_temp_y = mu_phy[x, y, s, a_temp]
                mu_temp_s = mu_phs[x, y, s, a_temp]
                mu_i = np.array(([mu_temp_x, mu_temp_y, mu_temp_s]))
                sj_xy = j%utils.num_obs_bins**2
                sj_s = j//utils.num_obs_bins**2
                sj_x = sj_xy%utils.num_obs_bins
                sj_y = sj_xy//utils.num_obs_bins
                s_j = np.array(([sj_x, sj_y, sj_s]))
                Wh[i, j] = alpha*np.exp(-0.5*(s_j-mu_i).T @ M @ (s_j - mu_i))
        Wh = alpha*Wh        

        # Approximately computing the maximum information gain
        K_rh = K_phx = K_phy = K_phs = np.zeros((K, K))
        #K_ph = np.zeros((K*m, K*m))
        for i in range(K):
            for j in range(K): 
                K_phx[i][j] = K_phy[i][j] = K_phs[i][j] = K_rh[i][j] = kr(z_h[i], z_h[j])

        """ for u in range(K*m):
            for v in range(K*m):
                K_ph[u][v] = kp(z_tilda_h[u][0:-1], z_tilda_h[u][-1], z_tilda_h[v][0:-1], z_tilda_h[v][-1])
 """
        #mig_rh = 1/2*math.log(np.linalg.det(np.identity(K)+ 1/args.lr*K_rh))
        #mig_phx = 1/2*math.log(np.linalg.det(np.identity(K) + 1/args.lp*K_phx))
        #mig_phy = 1/2*math.log(np.linalg.det(np.identity(K) + 1/args.lp*K_phy))
        #mig_phs = 1/2*math.log(np.linalg.det(np.identity(K) + 1/args.lp*K_phs))
        mig_rh = mig_phx = mig_phy = mig_phs = 1
        # This part is generating a bug: domain error. Maybe because the reward is not normalized to (0,1) so the matrix
        # is not invertible. Fixing 
        """ d = z[0].shape()
        mig_rh = math.log(K)**d
        mig_ph = math.log(K*m)**d """

        beta_rh = 1 + args.sdR/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_rh)
        beta_phx = 1 + args.sdP/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_phx)
        beta_phy = 1 + args.sdP/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_phy)
        beta_phs = 1 + args.sdP/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_phs)

        big_gamma_h = bv = Q_overline_h = Q_hat_h = pi_h = np.zeros([utils.num_obs_bins]*utils.state_dim+[utils.num_act_bins]*utils.action_dim)

        for x in np.ndindex((utils.num_obs_bins,)*utils.state_dim):
            for a in np.ndindex((utils.num_act_bins,)*utils.action_dim): 
                big_gamma_h[x][a] = beta_rh*sigma_rh[x][a] + HORIZON*np.sqrt(beta_phx**2 + beta_phy**2 + beta_phs**2)*np.sqrt(sigma_phx[x][a]**2 + sigma_phy[x][a]**2+sigma_phs[x][a]**2)
                V_temp = VHat[:,:,:, h].reshape((utils.num_obs_bins**utils.state_dim, 1))
                bv[x][a] = mu_rh[x][a] + 1/args.sigma*(np.matmul(Wh, V_temp).reshape([utils.num_obs_bins]*utils.state_dim)[x])
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
        prob[:,:,:,:,h-1] = pi_h
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
    parser.add_argument('--lamda', default=1)
    parser.add_argument('--var', default=0.3)
    parser.add_argument('--batch_size', default = 200)
    args = parser.parse_args()
    prob = main(args)
    np.save("gp_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved offline GP policy to", "gp_nt-{}_h-{}.npy".format(utils.num_trajectories, HORIZON))