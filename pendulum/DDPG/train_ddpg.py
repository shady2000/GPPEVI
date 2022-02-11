import gym
import torch
from torch import nn #needed for building neural networks
import torch.nn.functional as F #needed for activation functions
import torch.optim as opt #needed for optimisation
from tqdm import tqdm_notebook as tqdm
import random
from copy import copy, deepcopy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import settings
import models
import ddpg_buffer

print("Using torch version: {}".format(torch.__version__))

env = models.NormalizedEnv(gym.make(settings.ENV_NAME))
#env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

noise = models.OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

critic  = models.Critic(state_dim, action_dim).to(settings.device)
actor = models.Actor(state_dim, action_dim).to(settings.device)

actor.load_state_dict(torch.load("/media/Z/shun/storage/pendulum/model/best_model_pendulum_ddpg.pkl"))

target_critic  = models.Critic(state_dim, action_dim).to(settings.device)
target_actor = models.Actor(state_dim, action_dim).to(settings.device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    
q_optimizer  = opt.Adam(critic.parameters(),  lr=settings.LRC)#, weight_decay=0.01)
policy_optimizer = opt.Adam(actor.parameters(), lr=settings.LRA)

MSE = nn.MSELoss()

memory = ddpg_buffer.replayBuffer(settings.BUFFER_SIZE)

def subplot(R, P, Q, S):
    r = list(zip(*R))
    p = list(zip(*P))
    q = list(zip(*Q))
    s = list(zip(*S))
    clear_output(wait=True)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r') #row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b') #row=1, col=0
    ax[0, 1].plot(list(q[1]), list(q[0]), 'g') #row=0, col=1
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k') #row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Q loss')
    ax[1, 1].title.set_text('Max steps')
    plt.show()

plot_reward = []
plot_policy = []
plot_q = []
plot_steps = []


best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0
average_reward = 0
global_step = 0
#s = deepcopy(env.reset())

for episode in range(settings.MAX_EPISODES):
    #print(episode)
    s = deepcopy(env.reset())
    #noise.reset()

    ep_reward = 0.
    ep_q_value = 0.
    step=0

    for step in range(settings.MAX_STEPS):
        #loss=0
        global_step +=1
        settings.epsilon -= settings.epsilon_decay
        #actor.eval()
        a = actor.get_action(s) 
        #actor.train()

        a += noise()*max(0, settings.epsilon)
        a = np.clip(a, -1., 1.)
        s2, reward, terminal, info = env.step(a)


        memory.add(s, a, reward, terminal,s2)

        #keep adding experiences to the memory until there are at least minibatch size samples
        
        if memory.count() > settings.buffer_start:
            s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(settings.BATCH_SIZE)

            s_batch = torch.FloatTensor(s_batch).to(settings.device)
            a_batch = torch.FloatTensor(a_batch).to(settings.device)
            r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(settings.device)
            t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(settings.device)
            s2_batch = torch.FloatTensor(s2_batch).to(settings.device)
            
            
            #compute loss for critic
            a2_batch = target_actor(s2_batch)
            target_q = target_critic(s2_batch, a2_batch) #detach to avoid updating target
            y = r_batch + (1.0 - t_batch) * settings.GAMMA * target_q.detach()
            q = critic(s_batch, a_batch)
            
            q_optimizer.zero_grad()
            q_loss = MSE(q, y) #detach to avoid updating target
            q_loss.backward()
            q_optimizer.step()
            
            #compute loss for actor
            policy_optimizer.zero_grad()
            policy_loss = -critic(s_batch, actor(s_batch))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            policy_optimizer.step()
            
            #soft update of the frozen target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - settings.TAU) + param.data * settings.TAU
                )

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - settings.TAU) + param.data * settings.TAU
                )

        s = deepcopy(s2)
        ep_reward += reward


        #if terminal:
        #    noise.reset()
        #    break

    try:
        plot_reward.append([ep_reward, episode+1])
        plot_policy.append([policy_loss.data, episode+1])
        plot_q.append([q_loss.data, episode+1])
        plot_steps.append([step+1, episode+1])
    except:
        continue
    average_reward += ep_reward
    
    #if ep_reward > best_reward:

    #torch.save(actor.state_dict(), settings.model_path + '/best_model_pendulum_ddpg.pkl') #Save the actor model for future testing
    #print("Saved model to", settings.model_path + '/best_model_pendulum_ddpg.pkl')
    #best_reward = ep_reward
    #saved_reward = ep_reward
    #saved_ep = episode+1 

    if (episode % settings.PRINT_EVERY) == (settings.PRINT_EVERY-1):    # print every print_every episodes
        torch.save(actor.state_dict(), settings.model_path + '/best_model_actor_pendulum_ddpg_2.pkl') #Save the actor model for future testing
        print("Saved actor model to", settings.model_path + '/best_model_actor_model_pendulum_ddpg_2.pkl')
        torch.save(critic.state_dict(), settings.model_path + '/best_model_critic_pendulum_ddpg_2.pkl') #Save the actor model for future testing
        print("Saved critic model to", settings.model_path + '/best_model_critic_pendulum_ddpg_2.pkl')
        best_reward = ep_reward
        saved_reward = ep_reward
        saved_ep = episode+1
        subplot(plot_reward, plot_policy, plot_q, plot_steps)
        print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(settings.PRINT_EVERY) %
              (episode + 1, global_step, average_reward / settings.PRINT_EVERY))
        print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
        average_reward = 0 #reset average reward
