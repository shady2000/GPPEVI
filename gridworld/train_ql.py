import matplotlib.pyplot as plt
from env import BOARD_SIZE, GridWorld, HORIZON, ACTIONS
from agents import Q_Agent, RandomAgent, PEVIAgent, PQL_BCQ_Agent, FQIAgent
import numpy as np 
import utils 
import os

model_path = "/media/Z/shun/storage/gridworld/model/"

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

epsilon = 0.1

def train_ql(trials=20000):
    agent = Q_Agent()
    reward_per_episode = [] 
    all_reward_lists = [] 
    environment = GridWorld() 
    for trial in range(trials):
        # environment.reset()
        #print("The agent is playing trial number", trial)
        cumulative_reward = 0 
        reward_list = []
        h = 0
        while h < HORIZON: 
            old_state = environment.current_location
            action = agent.choose_action(old_state, h)
            #print("Chosen action", action)
            reward = environment.make_step(action)
            new_state = environment.current_location
            cumulative_reward += reward
            reward_list.append(cumulative_reward)
            h += 1
            agent.learn(old_state, reward, new_state, action) 
        reward_per_episode.append(cumulative_reward)
        all_reward_lists.append(reward_list)
        environment.reset()
        print("Cumulative reward of trial", trial, "is", cumulative_reward)
    #finished learning, now need to store the policy 
    prob = np.zeros([BOARD_SIZE]*2+[len(ACTIONS)])
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            state = (x, y)
            q_values_of_state = agent.q_table[state]
            maxValue = max(q_values_of_state.values())
            greedy_action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])\

            for a in range(len(ACTIONS)):
                if a == greedy_action: 
                    prob[x, y, a] = 1-agent.epsilon
                    #prob[x, y, a] = 1
                else: 
                    prob[x, y, a] = agent.epsilon/3
                    #prob[x, y, a] = 0
    np.save(model_path + "qltab2_nt-{}_h-{}".format(utils.num_trajectories, HORIZON), prob)
    print("Saved Q Learning Policy to", model_path + "qltab2_nt-{}_h-{}".format(utils.num_trajectories, HORIZON))
    sum_reward_list = []
    for h in range(HORIZON):
        cumul = 0
        for list in all_reward_lists: 
            cumul += list[h]
        sum_reward_list.append(cumul)
    avg_reward_list = [a/trials for a in sum_reward_list]
    return reward_per_episode, avg_reward_list

reward_per_episode_Q, avg_reward_list_Q = train_ql()
line_q, = plt.plot(moving_average(reward_per_episode_Q, 10), c = 'm', label = 'Q Agent') 
plt.legend()
plt.show()
