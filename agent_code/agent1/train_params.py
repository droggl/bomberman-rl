####### General training parameters #######

# Name of the model file
MODEL_NAME="testing.pt"

# creates a new model each time if True
RESET = False

# length of the feature vector
# TODO can we get rid of this?
FEATURE_LEN = 14


###### Q Learning parameters ######

# keep only ... last transitions
TRANSITION_HISTORY_SIZE = 400 

# record enemy transitions with probability ...
# RECORD_ENEMY_TRANSITIONS = 1.0  

# epsilon greedy
# EPSILON = 0.1

# softmax rho starting value
RHO_START = 2

# learning rate
GAMMA = 0.9


###### Gradient Boosting Regression parameters ######

# see sklearn reference
N_EST = 50
LEARN_RATE = 0.2
