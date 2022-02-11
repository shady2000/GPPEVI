from env import GridWorld, HORIZON
from agents import RandomAgent
import utils
import numpy as np
import os

data_path = "/media/Z/shun/storage/gridworld/dataset"
#collects data with random policy. Should consider mixing with suboptimal policy

def main(args): 
    K = utils.num_trajectories
    env = GridWorld()
    agent = RandomAgent()
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    available_actions = env.get_available_actions()
    for tau in range(K):
        print("Collecting trajectory", tau)
        h = 1
        while (h <= HORIZON):
            state = np.asarray(env.current_location)
            action = agent.choose_action(available_actions, h)
            reward = env.make_step(action)
            state_next = np.asarray(env.current_location)
            done = False
            dataset.add(state, action, reward, state_next, done)
            h = h+1
        env.reset()
    if not os.path.exists(data_path + "/random_dataset--nt-{}_h-{}".format(K, HORIZON)):
        os.makedirs(data_path + "/random_dataset--nt-{}_h-{}".format(K, HORIZON))
    dataset.save(data_path + "/random_dataset--nt-{}_h-{}".format(K, HORIZON))
    print("Offline data saved to", "dataset--nt-{}_h-{}".format(K, HORIZON))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
