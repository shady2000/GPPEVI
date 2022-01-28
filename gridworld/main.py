import matplotlib.pyplot as plt
from env import GridWorld, HORIZON
from agents import RandomAgent, PEVIAgent, PQL_BCQ_Agent, FQIAgent
import numpy as np 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def play(agent, trials=500, learn=False):
    reward_per_episode = [] 
    all_reward_lists = [] 
    environment = GridWorld() 
    for trial in range(trials):
        # environment.reset()
        print("The agent is playing trial number", trial)
        cumulative_reward = 0 
        reward_list = []
        h = 0
        while h < HORIZON: 
            old_state = environment.current_location
            action = agent.choose_action(old_state, h)
            reward = environment.make_step(action)
            new_state = environment.current_location
            cumulative_reward += reward
            reward_list.append(cumulative_reward)
            h += 1
            if learn == True:
                agent.learn(old_state, reward, new_state, action) 
        reward_per_episode.append(cumulative_reward)
        all_reward_lists.append(reward_list)

    sum_reward_list = []
    for h in range(HORIZON):
        cumul = 0
        for list in all_reward_lists: 
            cumul += list[h]
        sum_reward_list.append(cumul)
    avg_reward_list = [a/trials for a in sum_reward_list]
    return reward_per_episode, avg_reward_list

agentRand = RandomAgent()
reward_per_episode_rand, avg_reward_list_rand = play(agentRand)
agentPess = PEVIAgent("pevi_nt-500_h-40.npy")
reward_per_episode_pevi, avg_reward_list_pevi = play(agentPess)
agentPQLBCQ = PQL_BCQ_Agent('pqlbcq_policy.pkl')
reward_per_episode_pqlbcq, avg_reward_list_pqlbcq = play(agentPQLBCQ) 
agentFQI = FQIAgent('gpfqi_nt-500_h-40.npy')
reward_per_episode_fqi, avg_reward_list_fqi = play(agentFQI)

line_rand, = plt.plot(moving_average(avg_reward_list_rand, 1), c = 'r', label = 'Random Agent') 
line_pevi, = plt.plot(moving_average(avg_reward_list_pevi , 1), c = 'g', label = "PEVI (Modified)") 
line_pqlbcq, = plt.plot(moving_average(avg_reward_list_pqlbcq, 1), c = 'b', label = 'PQL BCQ Agent') 
line_pqlfqi, = plt.plot(moving_average(avg_reward_list_fqi, 1), c = 'k', label = 'FQI Agent')

plt.legend()
plt.show()

""" print(environment.agent_on_map())
"""