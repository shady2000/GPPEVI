# GPPEVI

In Gridworld:
    + BCQ.py, BEAR.py, train_pql.py are used for training PQL policy 
    + collect_data.py is used for collecting data 
    + env.py stores the configuration of the environment including of: 
        - HORIZON
        - BOARD_SIZE
        - ACTIONS Set
        - Transition noise 
        - Feature map phi 
        - Other environment configurations
    + gen_weight.py: generate random model parameters for Linear PEVI 
    + train_gp.py is used for training GP PEVI policy 
    + train_fqi.py is used for training FQI policy 
    + utils.py stores other configuration for the problem including of: 
        - Dataset class
        - Number of trajectories in dataset 
    + Environment configurations:
        - Current BOARD_SIZE: 20
        - Current HORIZON: 40 
        - Transition noise: 0.05 
        - Feature map: phi(state, action) = np.array([1/(x**2+1), 1/(y**2+1), 1/(action_index+1)])
        - Reward: 
            0 if at the half right-bottom corner 
            1 if at (1, 1)
            1/4 otherwise 
    + Then, the optimal strategy would be trying to reach the upper-left corner 

    + run python3 collect_data.py to collect data with random policy  
    + run python3 train_fqi.py to train fqi policy, similarly for train_pql.py, train_gp.py 
    + To configure location to save offline policy, go into each of the file for training offline policies


In Pendulum:
    + BCQ.py, BEAR.py, train_pql.py are used for training PQL policy 
    + collect_data.py is used for collecting data 
    + utils.py stores the configuration of the environment as well as the problem including of: 
        - HORIZON
        - BOARD_SIZE
        - ACTIONS Set
        - Transition noise 
        - Feature map phi 
        - Other environment configurations
        - Dataset class
        - Number of trajectories in dataset 
    + Forgot to setup Linear PEVI :))) 
    + train_gp.py is used for training GP PEVI policy 
    + train_fqi.py is used for training FQI policy 
    + Environment configurations:
        - Discretize state space into 10 states each dimension
        - Discretize action space into 10 actions each dimension 
        - State has 3 dimensions 
        - Action has 1 dimension 
        - 
        - Current HORIZON: 50 

    + run python3 collect_data.py to collect data with random policy  
    + run python3 train_fqi.py to train fqi policy, similarly for train_pql.py, train_gp.py 
    + To configure location to save offline policy, go into each of the file for training offline policies
