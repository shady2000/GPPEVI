import gym
import sys
sys.path.append('/media/Z/shun/offline-reinforcement-learning-2021/pendulum')
import utils
import os
import models
import torch 
from copy import deepcopy
device = torch.device("cpu")

model_path = "/media/Z/shun/storage/pendulum/model"
data_path = "/media/Z/shun/storage/pendulum/dataset"
model = "/best_model_actor_pendulum_ddpg.pkl" 

num_t = utils.num_trajectories
num_h = utils.HORIZON

env = gym.make(utils.env_name)
env.seed(1)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

ac = models.Actor(obs_dim, act_dim).to(device)
ac.load_state_dict(torch.load(model_path + model))

dataset = utils.Dataset(obs_dim, act_dim)

for tau in range(num_t):
    print("Collecting trajectory", tau)
    s_curr = deepcopy(env.reset())
    h = 1
    while h <= utils.HORIZON:
        a_curr = ac.get_action(s_curr)
        s_next, r_curr, done, _ = env.step(a_curr)
        dataset.add(s_curr, a_curr, r_curr, s_next, done)
        s_curr = s_next
        h += 1

if not os.path.exists(data_path + f"/ddpg_dataset--nt-{num_t}_h-{num_h}"):
    os.makedirs(data_path + f"/ddpg_dataset--nt-{num_t}_h-{num_h}")
dataset.save(data_path + f"/ddpg_dataset--nt-{num_t}_h-{num_h}")
print("Offline data saved to", data_path + f"/ddpg_dataset--nt-{num_t}_h-{num_h}")

