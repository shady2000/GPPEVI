from env import ACTIONS, GridWorld, HORIZON
from agents import RandomAgent
import utils
import numpy as np
import os

#collect data with qltab policy 
def main(args): 
    K = utils.num_trajectories
    env = GridWorld()
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)
    policy_all = np.load('./trained_policy/qltab2_nt-2000_h-40.npy')
    for tau in range(K):
        print("Collecting trajectory", tau)
        h = 1
        while (h <= HORIZON):
            state = np.asarray(env.current_location)
            #action = agent.choose_action(available_actions, h)
            policy_at_state = policy_all[state[0], state[1], :]
            action = np.random.choice(ACTIONS, p=policy_at_state)
            reward = env.make_step(action)
            state_next = np.asarray(env.current_location)
            done = False
            dataset.add(state, action, reward, state_next, done)
            h = h+1
        env.reset()
    if not os.path.exists("./qltab_dataset--nt-{}_h-{}".format(K, HORIZON)):
        os.makedirs("./qltab_dataset--nt-{}_h-{}".format(K, HORIZON))
    dataset.save("./qltab_dataset--nt-{}_h-{}".format(K, HORIZON))
    print("Offline data saved to", "qltab_dataset--nt-{}_h-{}".format(K, HORIZON))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)