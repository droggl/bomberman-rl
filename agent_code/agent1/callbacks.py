from curses import tparm
import os
import pickle
import random
import time
import numpy as np

from .features import *
from .online_gradient_boosting import online_gradient_boost_regressor as ogbr
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
    # if not training
    if not self.train:
        # try to load model
        if os.path.isfile(tparam.MODEL_NAME):
            self.logger.info("Loading model \"{}\" from saved state.".format(tparam.MODEL_NAME))
            with open(tparam.MODEL_NAME, "rb") as file:
                self.model_current = pickle.load(file)
        else:
            ...
            # TODO fix this mess 
        
        self.rho_play = 0.2

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
        Q_pred[i] = self.model_current[i].predict(X.reshape(1,-1))
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

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    player_pos = game_state["self"][3]
    others = game_state["others"]
    

    channels = []
    # local awareness navigation helper
    channels.append(not_traversible(field, bombs, explosion_map, others, player_pos))

    # bomb avoidance
    channels.append(survival_instinct(field, bombs, explosion_map, others, player_pos))

    # coin finder
    channels.append(coin_collector2(field, coins, player_pos))
    channels.append(crate_potential(field, player_pos))

    # concatenate channels
    concatenated_channels = np.concatenate(channels)
    # and return them as a vector
    return concatenated_channels
