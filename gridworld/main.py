from env import ACTIONS, GridWorld, HORIZON
from agents import RandomAgent
import utils
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import device
import tensorboardX
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=50)
parser.add_argument('--avg', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

figure_path = '/media/Z/shun/storage/gridworld/figures'
data_path = "/media/Z/shun/storage/gridworld/dataset"
model_path = "/media/Z/shun/storage/gridworld/model"

num_t = utils.num_trajectories
num_h = HORIZON

# Tensorboard & Logger
model_dir = f"{model_path}/nt-{num_t}_h-{num_h}_fromql_substract"

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def play(alg_name, policy="random", episodic=True):
    model_dir_name = model_dir+"_"+alg_name+"10FEB"
    txt_logger = utils.get_txt_logger(model_dir_name)
    tb_writer = tensorboardX.SummaryWriter(model_dir_name)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)
    txt_logger.info(f"Device: {device}\n")
    txt_logger.info(f"Model dir: {model_dir}\n")
    txt_logger.info(f"Start running algorithm {alg_name}")
    reward_per_episode = []
    all_reward_lists = []
    env = GridWorld()
    for trial in range(args.trials):
        txt_logger.info(f"Running trial {trial} of algorithm {alg_name}")
        h = 0
        cumulative_reward = 0 
        reward_list = []
        while (h < HORIZON):
            state = np.asarray(env.current_location)
            if policy == "random":
                action = np.random.choice(ACTIONS)
            else:
                # print(policy+".npy")
                policy_run = np.load(policy+".npy", allow_pickle=True)
                if episodic:
                    policy_at_state = policy_run[state[0], state[1], :, h]
                else:
                    policy_at_state = policy_run[state[0], state[1], :]
                policy_at_state = policy_at_state/np.sum(policy_at_state)
                action = np.random.choice(ACTIONS, p=policy_at_state)
            reward = env.make_step(action)
            cumulative_reward += reward

            # log for each step in episode
            reward_list.append(cumulative_reward)
            header_t = [f"reward_{trial}"]
            data_t = [cumulative_reward]
            for field, value in zip(header_t, data_t):
                tb_writer.add_scalar(field, value, h)

            h = h+1
        env.reset()
        reward_per_episode.append(cumulative_reward)
        all_reward_lists.append(reward_list)

        # log for each episode
        header = [f"reward_avg"]
        data = [cumulative_reward]
        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, trial)
    sum_reward_list = []
    for h in range(HORIZON):
        cumul = 0
        for list in all_reward_lists: 
            cumul += list[h]
        sum_reward_list.append(cumul)
    avg_reward_list = [a/args.trials for a in sum_reward_list]
    return reward_per_episode, avg_reward_list


# reward_per_episode_gppevi, avg_reward_list_gppevi = play("GPPEVI", f"{model_path}/gp_nt-{num_t}_h-{num_h}")
reward_per_episode_gppevi, avg_reward_list_gppevi = play("GPPEVI-aoma", f"{model_path}/gp_nt_aoma-{num_t}_h-{num_h}")
# reward_per_episode_ql, avg_reward_list_ql = play("Q-Learning", f"{model_path}/qltab2_nt-{num_t}_h-{num_h}", episodic=False)
reward_per_episode_ql, avg_reward_list_ql = play("Q-Learning", f"{model_path}/qltab2_nt-2000_h-40", episodic=False)

# reward_per_episode_rand, avg_reward_list_rand = play("Random_Policy")
# reward_per_episode_pevi_t, avg_reward_list_pevi_t = play("Linear_PEVI_True_Model", f"{model_path}/pevi_truephi_nt-{num_t}_h-{num_h}_fromqltab")
# reward_per_episode_pevi_t, avg_reward_list_pevi_t = play("Linear_PEVI_True_Model_random", f"{model_path}/pevi_true_nt-{num_t}_h-{num_h}_from_random")
# reward_per_episode_bellman, avg_reward_list_bellman = play("Bellman", f"bellman_nt-{num_t}", episodic=False)
# # reward_per_episode_pevi_w, avg_reward_list_pevi_w = play("Linear_PEVI_Wrong_Model", f"{model_path}/pevi_phi1_nt-{num_t}_h-{num_h}_fromqltab")

line_alg, = plt.plot(moving_average(avg_reward_list_gppevi, args.avg), c = 'b', label = 'GPPEVI Agent')
line_alg, = plt.plot(moving_average(avg_reward_list_ql, args.avg), c = 'm', label = 'Q Learning Agent')
line_pevi_t, = plt.plot(moving_average(avg_reward_list_pevi_t , args.avg), c = 'g', label = "Linear PEVI True Ft.Map")
# line_pevi_w, = plt.plot(moving_average(avg_reward_list_pevi_w , args.avg), c = 'k', label = "Linear PEVI Wrong Ft.Map")
line_bellman, = plt.plot(moving_average(avg_reward_list_bellman, args.avg), c = 'r', label = "Bellman")

""" line_alg, = plt.plot(moving_average(reward_per_episode_gppevi, args.avg), c = 'b', label = 'Our Alg')
line_alg, = plt.plot(moving_average(reward_per_episode_ql, args.avg), c = 'm', label = 'Q Learning Alg')
line_rand, = plt.plot(moving_average(reward_per_episode_rand, args.avg), c = 'r', label = 'Random Alg') 
line_pevi_t, = plt.plot(moving_average(reward_per_episode_pevi_t , args.avg), c = 'g', label = "Linear PEVI True Ft.Map")
line_pevi_w, = plt.plot(moving_average(reward_per_episode_pevi_w , args.avg), c = 'k', label = "Linear PEVI Wrong Ft.Map") """

plt.legend()
plt.savefig(figure_path + '/gridworld_eval.png')

# /media/Z/shun/storage/gridworld/model/pevi_true_nt-2000_h-40_from_random.npy
# /media/Z/shun/storage/gridworld/model/pevi_true_nt-2000_h-40_from_random.npy
