import gym
import utils
from utils import HORIZON
import os
from pg.models import ActorCritic
from gym.spaces import Box, Discrete
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    env = gym.make(utils.env_name)
    env.seed(1)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        discrete = True
        act_dim = env.action_space.n
    else:
        discrete = False
        act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, discrete).to(device)
    model.load_state_dict(torch.load(args.path))

    K = utils.num_trajectories
    dataset = utils.Dataset(utils.state_dim, utils.action_dim)

    for tau in range(K):
        print("Collecting trajectory", tau)
        #s_curr = utils.obs2bin(env.reset())
        s_curr = env.reset()
        h = 1
        while h <= HORIZON:
            a_curr, _, _ = model.step(torch.as_tensor(s_curr, dtype=torch.float32).to(device))
            s_next, r_curr, done, _ = env.step(a_curr)
            #s_next = utils.obs2bin(s_next)
            #a_curr = utils.act2bin(a_curr)
            dataset.add(s_curr, a_curr, r_curr, s_next, done)
            s_curr = s_next
            h += 1
    if not os.path.exists("./pg_dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins)):
        os.makedirs("./pg_dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    dataset.save("./pg_dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))
    print("Offline data saved to", "pg_dataset--nt-{}_h-{}_s-{}_a-{}".format(K, HORIZON, utils.num_obs_bins, utils.num_act_bins))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./pg/best_model_Pendulum.pt', type=str, help="Path to the model weights")
    args = parser.parse_args()
    main(args)