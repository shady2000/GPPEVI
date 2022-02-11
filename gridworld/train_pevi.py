from email import header
import math
from env import BOARD_SIZE, HORIZON, ACTIONS, phi
import numpy as np
from env import GridWorld
import utils 
from utils import device
import tensorboardX
import sys
ps = []
data_path = "/media/Z/shun/storage/gridworld/dataset"
model_path = "/media/Z/shun/storage/gridworld/model"
# policy_name = f""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--exp', type=float, default=0.05)
#parser.add_argument('--lamb', type=float, default=1)
#parser.add_argument('--beta', type=float)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

# Tensorboard & Logger
# model_dir = f"{model_path}/pevi_phi1_nt-{utils.num_trajectories}_h-{HORIZON}_fromql_substract"
model_dir = f"{model_path}/pevi_true_nt-{utils.num_trajectories}_h-{HORIZON}_from_random"
txt_logger = utils.get_txt_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)
txt_logger.info(f"Device: {device}\n")
txt_logger.info(f"Model dir: {model_dir}\n")

# Load env
dataset = utils.Dataset(utils.state_dim, utils.action_dim)
# dataset.load(data_path + "/qltab_dataset--nt-2000_h-40")
dataset.load(data_path + f"/random_dataset--nt-{utils.num_trajectories}_h-{HORIZON}")
env = GridWorld()
K = utils.num_trajectories
d = phi(env.current_location, ACTIONS[0]).shape[0]
prob = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)]+[HORIZON])
VHat = np.zeros([BOARD_SIZE]*utils.state_dim+[HORIZON+1]) 
states = dataset.state[0:K*HORIZON,:].reshape((K, HORIZON, utils.state_dim))
actions = dataset.action[0:K*HORIZON,:].reshape((K, HORIZON, utils.action_dim))
rewards = dataset.reward[0:K*HORIZON].reshape((K, HORIZON))

#Hyperparameters 
exp = args.exp
lamb = 1
shrink_rate = 0.5
beta_init = 1
#epsilon = args.epsilon
#xi = math.log(2*d*HORIZON*K/epsilon)
#lamb = 1
#c = 20
#beta = c*d*HORIZON*math.sqrt(xi)
K = utils.num_trajectories 
VHat = np.zeros([BOARD_SIZE, BOARD_SIZE, HORIZON+1]) 
state = env.current_location
d = phi(state, ACTIONS[0]).shape[0]

def train():
    num_horizon = 0
    for h in list(range(1, HORIZON+1))[::-1]:

        print("Running at period", h)
        states_h, actions_h, rewards_h = states[:, h-1, :], actions[:, h-1, :], rewards[:, h-1]

        header = ["state", "action", "reward"]
        data = [states_h, actions_h, rewards_h]

        big_lambd = lamb*np.identity(d)
        var = 0
        for tau in range(K):
            s_tau_h, a_tau_h, r_tau_h = states_h[tau], actions_h[tau], rewards_h[tau]
            big_lambd = big_lambd + np.matmul(phi(s_tau_h, a_tau_h), phi(s_tau_h, a_tau_h).T)
            id = tuple(s_tau_h.astype(np.int32).reshape(1, -1)[0])
            var = var + phi(s_tau_h, a_tau_h)*(r_tau_h + VHat[id][h])
            # header_K = ["tau", "r_tau_h"]
            # data_K = [tau, r_tau_h.item()]
            # # print(type(s_tau_h), type(a_tau_h), type(r_tau_h.item()), type(big_lambd), type(var))
            # for field, value in zip(header_K, data_K):
            #     tb_writer.add_scalar(field, value, tau)
        w_hat_h = np.linalg.inv(big_lambd).dot(var)

        Big_Gamma_h = Q_Overline_h = Q_hat_h = pi_h = np.zeros([BOARD_SIZE]*utils.state_dim+[len(ACTIONS)])
        index_x = 0
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
                    #pi_h[x][a] = 1 - args.exp
                    pi_h[x][a] = 1
                else:
                    #pi_h[x][a] = args.exp/(len(ACTIONS))
                    pi_h[x][a] = 0
            index_x+=1
        for x in np.ndindex((BOARD_SIZE,)*utils.state_dim):
            sum = np.sum(pi_h[x][:])
            for a in range(len(ACTIONS)):
                pi_h[x][a] = pi_h[x][a]/sum
            VHat[x][h-1] = np.dot(Q_hat_h[x][:], pi_h[x][:])
        prob[:, :, :, h-1] = pi_h
        # header_678 = ["Q_Overline_h_"+str(a) for a in range(len(ACTIONS))]
        # header_678+= ["Q_hat_h_"+str(a) for a in range(len(ACTIONS))]
        # header_678+= ["Big_Gamma_h_"+str(a) for a in range(len(ACTIONS))]
        # # print(type(Q_Overline_h[0][0]), type(Q_hat_h[0][0]), type(Big_Gamma_h[0][0]))
        # # print(Q_Overline_h[0][0].shape, Q_hat_h[0][0].shape, Big_Gamma_h[0][0].shape)
        # # print(Q_Overline_h[0][0][a].item())
        # data_678 = [Q_Overline_h[0][0][a].item() for a in range(len(ACTIONS))]
        # data_678+= [Q_hat_h[0][0][a].item() for a in range(len(ACTIONS))]
        # data_678+= [Big_Gamma_h[0][0][a].item() for a in range(len(ACTIONS)) ]
        # # print(type(x))
        # for field, value in zip(header_678, data_678):
        #     tb_writer.add_scalar(field, value, num_horizon)
        num_horizon+=1

prob = train()
np.save(model_dir, prob)
txt_logger.info(f"Saved offline policy to {model_dir}.npy")
