from curses import tparm
import os
import pickle
import random
import time
from sklearn.ensemble import GradientBoostingRegressor as GBR

import numpy as np
from .aux import coin_collector, coin_collector2, survival_instinct, traversible
import agent_code.agent1.train_params as tparam


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # set up new model
    if tparam.RESET == True or not os.path.isfile(tparam.MODEL_NAME):
        self.logger.info("Setting up model \"{}\" from scratch.".format(tparam.MODEL_NAME))

        # init model with trivial values
        # TODO evaluate init
        dummy_X = np.zeros(tparam.FEATURE_LEN).reshape((1,-1))
        dummy_y = np.zeros(1)
        self.model = []
        for i in range(6):
            self.model.append(GBR(n_estimators=tparam.N_EST, learning_rate=tparam.LEARN_RATE).fit(dummy_X, dummy_y))
    # use existing model 
    else: 
        self.logger.info("Loading model \"{}\" from saved state.".format(tparam.MODEL_NAME))
        with open(tparam.MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)

    self.rho_play = 0.1

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # transfer state to features
    t_0 = time.time()
    X = state_to_features(game_state)
    t_1 = time.time()
    
    # Query model
    Q_pred = np.empty(6)
    for i in range(6):
        Q_pred[i] = self.model[i].predict(X.reshape(1,-1))
    t_2 = time.time()

    self.logger.info("Feature / Model / Total (ms): {}  {}  {}".format((t_1-t_0) * 1000, (t_2-t_1) * 1000, (t_2-t_0) * 1000))

    # train using softmax
    # TODO improved epsilon-greedy from lecture?
    # TODO implement rho annealing
    if self.train:
        Q_pred = np.exp(Q_pred / self.rho)
        p_decision = Q_pred / np.sum(Q_pred)
        self.logger.debug("Softmax choice, p = " + np.array_str(p_decision))
        # print(p_decision)
        return np.random.choice(ACTIONS, p=p_decision)
    # play using greedy
    else: 
        Q_pred = np.exp(Q_pred / self.rho_play)
        p_decision = Q_pred / np.sum(Q_pred)
        self.logger.info("Softmax choice, p = " + np.array_str(p_decision))
        return np.random.choice(ACTIONS, p=p_decision)


def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    channels = []
    # local awareness navigation helper
    channels.append(traversible(game_state["field"], game_state["bombs"], game_state["explosion_map"], game_state["self"][3]))

    # bomb avoidance
    channels.append(survival_instinct(game_state["field"], game_state["bombs"], game_state["explosion_map"], game_state["self"][3]))

    # coin finder
    channels.append(coin_collector2(game_state["field"], game_state["coins"], game_state["self"][3]))
    # channels.append(coin_collector(game_state["field"], game_state["coins"], game_state["self"][3]))

    # concatenate channels
    concatenated_channels = np.concatenate(channels)
    # and return them as a vector
    return concatenated_channels
