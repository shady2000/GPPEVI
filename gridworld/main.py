from env import ACTIONS, GridWorld, HORIZON
from agents import RandomAgent
import utils
import numpy as np
import os
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def play(alg_name, policy="random", episodic=True):
    print("Start running algorithm", alg_name)
    reward_per_episode = []
    all_reward_lists = []
    env = GridWorld()
    for trial in range(args.trials):
        print("Running trial", trial, "of algorithm", alg_name)
        h = 0
        cumulative_reward = 0 
        reward_list = []
        while (h < HORIZON):
            state = np.asarray(env.current_location)
            if policy == "random":
                action = np.random.choice(ACTIONS)
            else: 
                policy_run = np.load(policy)
                if episodic:
                    policy_at_state = policy_run[state[0], state[1], :, h]
                else:
                    policy_at_state = policy_run[state[0], state[1], :]
                action = np.random.choice(ACTIONS, p=policy_at_state)
            reward = env.make_step(action)
            cumulative_reward += reward
            reward_list.append(cumulative_reward)
            h = h+1
        env.reset()
        reward_per_episode.append(cumulative_reward)
        all_reward_lists.append(reward_list)
    sum_reward_list = []
    for h in range(HORIZON):
        cumul = 0
        for list in all_reward_lists: 
            cumul += list[h]
        sum_reward_list.append(cumul)
    avg_reward_list = [a/args.trials for a in sum_reward_list]
    return reward_per_episode, avg_reward_list

def main(args): 
    reward_per_episode_gppevi, avg_reward_list_gppevi = play("GPPEVI", "./trained_policies/gp_nt-2000_h-40.npy")
    reward_per_episode_ql, avg_reward_list_ql = play("Q-Learning", "./trained_policies/qltab_deterministic_nt-2000_h-40.npy", episodic=False)
    reward_per_episode_rand, avg_reward_list_rand = play("Random Policy")
    reward_per_episode_pevi_t, avg_reward_list_pevi_t = play("Linear PEVI True Model", "./trained_policies/pevi_truephi_nt-2000_h-40_fromqltab.npy")
    reward_per_episode_pevi_w, avg_reward_list_pevi_w = play("Linear PEVI Wrong Model", "./trained_policies/pevi_wrongphi_nt-2000_h-40_fromqltab.npy")

    line_alg, = plt.plot(moving_average(avg_reward_list_gppevi, args.avg), c = 'b', label = 'GPPEVI Agent')
    line_alg, = plt.plot(moving_average(avg_reward_list_ql, args.avg), c = 'm', label = 'Q Learning Agent')
    line_rand, = plt.plot(moving_average(avg_reward_list_rand, args.avg), c = 'r', label = 'Random Agent') 
    line_pevi_t, = plt.plot(moving_average(avg_reward_list_pevi_t , args.avg), c = 'g', label = "Linear PEVI True Ft.Map")
    line_pevi_w, = plt.plot(moving_average(avg_reward_list_pevi_w , args.avg), c = 'k', label = "Linear PEVI Wrong Ft.Map")

    """ line_alg, = plt.plot(moving_average(reward_per_episode_gppevi, args.avg), c = 'b', label = 'Our Alg')
    line_alg, = plt.plot(moving_average(reward_per_episode_ql, args.avg), c = 'm', label = 'Q Learning Alg')
    line_rand, = plt.plot(moving_average(reward_per_episode_rand, args.avg), c = 'r', label = 'Random Alg') 
    line_pevi_t, = plt.plot(moving_average(reward_per_episode_pevi_t , args.avg), c = 'g', label = "Linear PEVI True Ft.Map")
    line_pevi_w, = plt.plot(moving_average(reward_per_episode_pevi_w , args.avg), c = 'k', label = "Linear PEVI Wrong Ft.Map") """

    plt.legend()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--avg', type=int, default=1)
    args = parser.parse_args()
    main(args)