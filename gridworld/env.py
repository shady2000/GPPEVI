import numpy as np
import pandas as pd

HORIZON = 40
BOARD_SIZE = 20
ACTIONS = [0, 1, 2, 3]
trans_noise = 0.05

def phi(state, action): 
    x = state[0]
    y = state[1]
    action_index = ACTIONS.index(action)
    return np.array([1/(x**2+1), 1/(y**2+1), 1/(action_index+1)])

class GridWorld:
    def __init__(self):
        # Set information about the gridworld
        self.height = BOARD_SIZE
        self.width = BOARD_SIZE
        self.grid = np.zeros(( self.height, self.width)) - 1
        
        # Set random start location for the agent
        self.current_location = (np.random.randint(0,BOARD_SIZE), np.random.randint(0,BOARD_SIZE))
        self.actions = ACTIONS

        # Set locations for special states if special states are needed
        """ 
        self.bomb_location = (1,3)
        self.gold_location = (0,3)
        self.terminal_states = [ self.bomb_location, self.gold_location] 
        """
        
        # Set grid rewards for special cells
        """ 
        self.grid[ self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[ self.gold_location[0], self.gold_location[1]] = 10 
        """
    
    def reset(self): 
        #self.current_location = (np.random.randint(0,BOARD_SIZE), np.random.randint(0,BOARD_SIZE))
        self.current_location = (BOARD_SIZE, BOARD_SIZE)

    def get_available_actions(self):
        return self.actions
    
    def agent_on_map(self):
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid
    
    def get_reward(self, current_location, action):
        """Returns the reward for an input position"""
        x = current_location[0]
        y = current_location[1]
        p = phi(current_location, action)
        reward_noise = np.random.normal(scale = 0.01)
        w_df = pd.read_csv("rw.csv")
        w_array = w_df.to_numpy()[:,1]/(BOARD_SIZE+BOARD_SIZE+len(ACTIONS))
        raw_reward = np.dot(w_array, p)
        if (x >= BOARD_SIZE/2 and y >= BOARD_SIZE/2):
            return 0
        elif (x < BOARD_SIZE/2 and y < BOARD_SIZE/2 and x>1 and y >1):
            return 1/4
        elif (x == 1 and y == 1):
            return 1
        else:
            return np.clip(0, 1, raw_reward + reward_noise)

    def make_step(self, action):
        last_location = self.current_location
        reward = self.get_reward(last_location, action)

        if np.random.uniform(0,1) < 1 - trans_noise:
            if action == 0:
                if last_location[0] != 0:
                    self.current_location = (last_location[0] - 1, last_location[1])
            
            elif action == 1:
                if last_location[0] != self.height - 1:
                    self.current_location = (last_location[0] + 1, last_location[1])
                
            elif action == 2:
                if last_location[1] != 0:
                    self.current_location = (last_location[0], last_location[1] - 1)

            elif action == 3:
                if last_location[1] != self.width - 1:
                    self.current_location = (last_location[0], last_location[1] + 1)

        else: 
            raw_available_next_states = [(last_location[0] - 1, last_location[1]), (last_location[0] + 1, last_location[1]),\
                    (last_location[0], last_location[1] - 1), (last_location[0], last_location[1] + 1)]
            raw_next_state = raw_available_next_states[np.random.randint(0, len(raw_available_next_states))]
            valid_x = np.clip(raw_next_state[0], 0, BOARD_SIZE-1)
            valid_y = np.clip(raw_next_state[1], 0, BOARD_SIZE-1)
            self.current_location = (valid_x, valid_y)
        return reward
