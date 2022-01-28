import gym 
import numpy as np
import matplotlib.pyplot as plt
from utils import env_name, HORIZON
from agents import FQIAgent, PQL_BCQ_Agent, RandomAgent, PEVIAgent

env = gym.make(env_name)
env.seed(1)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def play(agent, trials=1):
    reward_per_episode = []
    all_reward_lists = []  
    for trial in range(trials):
        print("The agent is playing trial number", trial)
        cumulative_reward = 0 
        reward_list = []
        h = 0
        s = env.reset()
        while h < HORIZON: 
            #env.render()
            a = agent.choose_action(s, h)
            s_next, r, _, _ = env.step(a)
            s = s_next
            r += 15
            cumulative_reward += r
            reward_list.append(cumulative_reward)
            h += 1
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

# need to specify the location for policy file for each agent 
agentRand = RandomAgent()
reward_per_episode_rand, avg_reward_list_rand = play(agentRand)
#agentPess1 = PEVIAgent("pevi_nt-500_h-50_s-10_a-10.npy")
#reward_per_episode_pevi_1, avg_reward_list_pevi_1 = play(agentPess1)
#agentPQLBCQ = PQL_BCQ_Agent('pqlbcq_policy.pkl')
#reward_per_episode_pqlbcq, avg_reward_list_pqlbcq = play(agentPQLBCQ) 
agentFQI = FQIAgent('gpfqi_nt-500_s-10_a-10.npy')

reward_per_episode_fqi, avg_reward_list_fqi = play(agentFQI)  
# plotting the performance 
line_rand, = plt.plot(moving_average(avg_reward_list_rand, 1), c = 'r', label = 'Random Agent')
#line_pevi_1, = plt.plot(moving_average(avg_reward_list_pevi_1 , 1), c = 'g', label = "PEVI (Modified)")
#line_pqlbcq, = plt.plot(moving_average(avg_reward_list_pqlbcq, 1), c = 'b', label = 'PQL BCQ Agent') 
line_pqlfqi, = plt.plot(moving_average(avg_reward_list_fqi, 1), c = 'k', label = 'FQI Agent')

plt.legend()
plt.show()