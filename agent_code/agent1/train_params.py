####### General training parameters #######

# Name of the model file
MODEL_NAME="testing.pt"

# creates a new model each time if True
RESET = False

# length of the feature vector
# TODO can we get rid of this?
FEATURE_LEN = 20


###### Q Learning parameters ######

# transition buffer size
TRANSITION_HISTORY_SIZE = 400

# replace buffer for each iteration
BUFFER_CLEAR = True

# Q steps (normal Q learning = 1)
Q_STEPS = 1

# Q rate
Q_RATE = 0.9

# epsilon greedy
EPSILON = 0.2

# softmax rho starting value
RHO_START = 2


###### Gradient Boosting Regression parameters ######

# number of estimators used
CYCLE_TIME = 10

# learning rate
GB_RATE = 0.2

### Weak Estimator parameters ###

# see sklearn reference
WEAK_N_EST = 60
WEAK_RATE = 0.1
