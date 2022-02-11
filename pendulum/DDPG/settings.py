BUFFER_SIZE=1000000
BATCH_SIZE=64
GAMMA=0.99
TAU=0.001       #Target Network HyperParameters Update rate
LRA=0.0001      #LEARNING RATE ACTOR
LRC=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers

MAX_EPISODES=5000 #number of episodes of the training
MAX_STEPS=100    
buffer_start = 100 #initial warmup without training
epsilon = 1
epsilon_decay = 1./100000 
PRINT_EVERY = 10 #Print info about average reward every PRINT_EVERY

ENV_NAME = "Pendulum-v0"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ("cpu")
model_path = "/media/Z/shun/storage/pendulum/model"
