from cmath import tau
import math
import numpy as np
import utils
from env import BOARD_SIZE, HORIZON, ACTIONS
from env import GridWorld
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
import time 
import random
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
    dataset.load("./qltab_dataset--nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    env = GridWorld()
    K = utils.num_trajectories
    d = utils.phi(env.current_location, ACTIONS[0]).shape[0]
    prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]+[HORIZON])
    VHat = np.zeros([BOARD_SIZE]*utils.state_dim+[HORIZON+1]) 
    states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
    rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))
    next_states = dataset.next_state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
    Z_all = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), 3))
    m = len(env.current_location)
    #Z_tilda_all = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m, 4))
    for x in np.ndindex((BOARD_SIZE, BOARD_SIZE)):
        for a in range(len(ACTIONS)):
            Z_all[x][a][:] = np.array([x[0], x[1], ACTIONS[a]])
    #for x in np.ndindex((BOARD_SIZE, BOARD_SIZE)):
    #    for a in range(len(ACTIONS)): 
    #        for i in range(m): 
    #            Z_tilda_all[x][a][i][:] = np.array([x[0], x[1], ACTIONS[a], i])

    Z_all = Z_all.reshape(-1, 3)
    #Z_tilda_all = Z_tilda_all.reshape(-1,4)
    #Every Z_tilda_all or z_tilda_h appears in this code is for the setting of Online Learning paper
    return dataset, env, prob, VHat, states, next_states, actions, rewards, K, d, m, Z_all#, Z_tilda_all

def main(args):
    _, env, prob, VHat, states, next_states, actions, rewards, K, d, m, Z_all = init()

    kernel = ConstantKernel(1.0,constant_value_bounds=(1e-2, 10)) * RBF(length_scale=1.0,length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,1))
    gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_px = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_py = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    for h in list(range(1, HORIZON+1))[::-1]: 
        print("Running at period h = ", h)
        states_h, actions_h, rewards_h, next_states_h = states[:, h-1, :], actions[:, h-1, :], rewards[:, h-1], next_states[:, h-1, :]
        states_xh = states_h[:,0].reshape(-1, 1)
        states_yh = states_h[:,1].reshape(-1,1)   
 
        z_h = np.concatenate((states_h, actions_h), axis=1)
        z_xh = np.concatenate((states_xh, actions_h), axis=1)
        z_yh = np.concatenate((states_yh, actions_h), axis=1)

        #z_tilda_h = np.zeros((K*m, 4))
        #for id in range(K*m):
        #    if id<=K-1:
        #        z_tilda_h[id] = np.array([states_h[id][0], states_h[id][1], actions_h[id], 1], dtype=object)
        #    else: 
        #        z_tilda_h[id] = np.array([states_h[id-K][0], states_h[id-K][1], actions_h[id-K], 2], dtype=object)
        #S_h = np.ravel(next_states_h, order='F')
        #kernel = ConstantKernel(1.0,constant_value_bounds=(1e-2, 10)) * RBF(length_scale=1.0,length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,1))

        random_idx = random.sample(range(utils.num_trajectories), args.batch_size)
        start_time = time.time()
        gp_r.fit(z_h[random_idx], rewards_h[random_idx])
        mu_rh, sigma_rh = gp_r.predict(Z_all, return_std=True)

        #training GP_P with the setting of online learning paper
        #gp_p = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        #gp_p.fit(z_tilda_h, S_h) 
        #mu_ph, sigma_ph = gp_p.predict(Z_tilda_all, return_std=True)
        #mu_ph = mu_ph.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m))
        #sigma_ph = sigma_ph.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS), m)) 
        
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #gp_r = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        #gp_px = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        #gp_py = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

        gp_px.fit(z_xh[random_idx], next_states_h[random_idx, 0])
        mu_phx, sigma_phx = gp_px.predict(Z_all[:,[0, 2]], return_std=True)

        gp_py.fit(z_yh[random_idx], next_states_h[random_idx, 1])
        mu_phy, sigma_phy = gp_py.predict(Z_all[:,[1, 2]], return_std=True)        
        
        mu_rh = mu_rh.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        sigma_rh = sigma_rh.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        mu_phx = mu_phx.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        sigma_phx = sigma_phx.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        mu_phy = mu_phy.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        sigma_phy = sigma_phy.reshape((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))
        
        print("Time taken for all GP update is", time.time()-start_time)
        start_time = time.time()

        tau = args.lamda*np.eye(2)
        cov = np.array([[args.var, 0], [0, args.var]])
        #mean = np.array([mu_phx, mu_phy])
        M = np.linalg.inv(tau+cov)
        alpha = np.sqrt(np.linalg.det(M)*args.lamda)*args.sigma
        Wh = np.zeros((BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE))
        for i in range(BOARD_SIZE*BOARD_SIZE):
            for j in range(BOARD_SIZE*BOARD_SIZE):
                x = i%BOARD_SIZE
                y = i//BOARD_SIZE
                a_temp = np.argmax(prob[x,y,:,h-1])
                mu_temp_x = mu_phx[x, y, a_temp] 
                mu_temp_y = mu_phy[x, y, a_temp]
                mu_i = np.array(([mu_temp_x, mu_temp_y]))
                sj_x = j%BOARD_SIZE
                sj_y = j//BOARD_SIZE
                s_j = np.array(([sj_x, sj_y]))
                Wh[i, j] = alpha*np.exp(-0.5*(s_j-mu_i).T @ M @ (s_j - mu_i))
        Wh = alpha*Wh
        
        K_rh = K_phx = K_phy = np.zeros((K, K))
        for i in range(K):
            for j in range(K): 
                K_phx[i][j] = K_phy[i][j] = K_rh[i][j] = kr(z_h[i], z_h[j])

        #K_ph = np.zeros((K*m, K*m))
        #for u in range(K*m):
        #    for v in range(K*m):
        #        K_ph[u][v] = kp(z_tilda_h[u][0:-1], z_tilda_h[u][-1], z_tilda_h[v][0:-1], z_tilda_h[v][-1])

        mig_rh = 1/2*math.log(np.linalg.det(np.identity(K)+ 1/args.lr*K_rh))
        mig_phx = 1/2*math.log(np.linalg.det(np.identity(K) + 1/args.lp*K_phx))
        mig_phy = 1/2*math.log(np.linalg.det(np.identity(K) + 1/args.lp*K_phy))

        #d = z[0].shape()
        #mig_rh = math.log(K)**d
        #mig_ph = math.log(K*m)**d

        beta_rh = 1 + args.sdR/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_rh)
        beta_phx = 1 + args.sdP/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_phx)
        beta_phy = 1 + args.sdP/math.sqrt(HORIZON)*math.sqrt(2*math.log(3/args.delta) + mig_phy)

        big_gamma_h = bv = Q_overline_h = Q_hat_h = pi_h = np.zeros((BOARD_SIZE, BOARD_SIZE, len(ACTIONS)))

        for x in np.ndindex((BOARD_SIZE,)*utils.state_dim):
            for a in range(len(ACTIONS)): 
                #big_gamma_h[x][a] = beta_rh*sigma_rh[x][a] + HORIZON*beta_ph*np.sqrt(sigma_ph[x][a][0]**2 + sigma_ph[x][a][1]**2)
                #bv[x][a] = mu_rh[x][a] + np.sum(VHat[:, :, h]*np.sqrt(mu_ph[:,:,a,0]**2 + mu_ph[:,:,a,1]**2))
                big_gamma_h[x][a] = beta_rh*sigma_rh[x][a] + HORIZON*np.sqrt(beta_phx**2+beta_phy**2)*np.sqrt(sigma_phx[x][a]**2 + sigma_phy[x][a]**2)
                V_temp = VHat[:,:,h].reshape((BOARD_SIZE*BOARD_SIZE, 1))
                bv[x][a] = mu_rh[x][a] + 1/args.sigma*(np.matmul(Wh, V_temp).reshape((BOARD_SIZE, BOARD_SIZE))[x])
                big_gamma_h = np.maximum(0.01, big_gamma_h)
                Q_overline_h[x][a] = np.random.normal(loc = bv[x][a], scale = big_gamma_h[x][a])
                Q_hat_h = np.clip(0, HORIZON-h+1, Q_overline_h)
            for a in range(len(ACTIONS)):
                if (a == np.argmax(Q_hat_h[x][:])):
                    #pi_h[x][a] = 1 - args.exp (exploration strategy)
                    pi_h[x][a] = 1
                else:
                    #pi_h[x][a] = args.exp/(len(ACTIONS))
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
    parser.add_argument('--lamda', default=1)
    parser.add_argument('--var', default=0.3)
    parser.add_argument('--batch_size', default = 200)
    args = parser.parse_args()
    prob = main(args)
    np.save("gp_verify_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved offline GP policy to", "gp_verify_nt-{}_h-{}.npy".format(utils.num_trajectories, HORIZON))