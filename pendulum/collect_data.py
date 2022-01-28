import gym
import utils
from utils import HORIZON
import pandas as pd
import os

#collect offline data with random policy. Should consider with mixing with suboptimal policy 

def main(args):
    env = gym.make(utils.env_name)
    K = utils.num_trajectories
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    for tau in range(K):
        print("Collecting trajectory", tau)
        s_curr = utils.obs2bin(env.reset())
        h = 1
        while h <= HORIZON:
            a_curr = env.action_space.sample()
            s_next, r_curr, done, _ = env.step(a_curr)
            s_next = utils.obs2bin(s_next)
            a_curr = utils.act2bin(a_curr)
            dataset.add(s_curr, a_curr, r_curr, s_next, done)
            s_curr = s_next
            h += 1
    if not os.path.exists("./dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins)):
        os.makedirs("./dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    dataset.save("./dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    print("Offline data saved to", "dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)