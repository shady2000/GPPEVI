import pandas as pd 
from gridworld.train_pevi import phi
from env import GridWorld, ACTIONS
from agents import RandomAgent
import numpy as np 

# generates parameter for environment
def gen_weight(): 
    env = GridWorld()
    agent = RandomAgent()
    s = env.current_location
    d = len(phi(s, ACTIONS[0]))
    w = np.random.uniform(size = d)
    df = pd.DataFrame(w) 
    df.to_csv("rw.csv")

gen_weight()


