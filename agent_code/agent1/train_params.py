####### General training parameters #######

# Name of the model file
from tkinter import TRUE


MODEL_NAME="testing.pt"

# creates a new model each time if True
RESET = False

# length of the feature vector
# TODO can we get rid of this?
FEATURE_LEN = 31


###### Q Learning parameters ######

# transition buffer size
TRANSITION_HISTORY_SIZE = 20000

# replace buffer for each iteration
BUFFER_CLEAR = True

# Q steps (normal Q learning = 1)
Q_STEPS = 1

# Q rate
Q_RATE = 0.9

# epsilon greedy
EPSILON = 0.01

# softmax rho starting value
RHO_START = 1


###### Gradient Boosting Regression parameters ######

# number model updates before replacement
CYCLE_TIME = 1

## see sklearn reference

# number of estimators
N_EST = 300

# learning rate 
GB_RATE = 0.1

#weak estimator max depth
MAX_DEPTH = 10
