import os
import pickle
import random
from time import time

from agent_code.deepQuapsel.stat_recorder import stat_recorder

import agent_code.deepQuapsel.dql_params as params
from keras.models import load_model

import numpy as np
import tensorflow as tf
from agent_code.deepQuapsel.utils import create_dql_model


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
    tf.random.set_seed(1)
    random.seed(1)
    np.random.seed(1)

    self.timing = stat_recorder("./logs/timing.log")

    self.epsilon = params.EPSILON_START

    self.model = create_dql_model()
    self.target_model = create_dql_model() # gets updated every n episodes, used for prediction, determine future q values
    if (self.train and params.RESET) or not os.path.isfile(params.MODELNAME):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model weights from saved state.")
        self.model.load_weights(params.MODELNAME)
    self.logger.info(self.model.summary())
    self.target_model.set_weights(self.model.get_weights())


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_index = np.random.choice(range(6), p=[.2, .2, .2, .2, .1, .1])
    else:
        t1 = time()
        features = state_to_features(game_state)

        if params.TIMING_FEATURES:
            self.timing.write(f"state to features: {round((time()-t1) * 1000,3)}ms")

        t2 = time()
        q_values = self.model.predict(np.array(features).reshape(-1, *features.shape))[0]
        self.logger.debug(f"Q values: {q_values}")
        # Deciding by argmax
        action_index = np.argmax(q_values)
        # Deciding by softmax
        # p_decision = np.exp(q_values / params.RHO_PLAY)
        # p_decision = p_decision / np.sum(p_decision)
        # self.logger.info("Deciding with p = " + np.array_str(p_decision))
        # action_index = np.random.choice(range(6), p=p_decision)

        if params.TIMING_PREDICT:
            self.timing.write(f"predict: {round((time()-t2) * 1000)}ms")

    return ACTIONS[action_index]



def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array of 17x17x4: last dimension maps as follows:       possible values:
        0: free (0), crates (0.5), stone walls (1)                      [0, 0.5, 1]
        1: other players (0.5) + player (1)                             [0, 0.5, 1]
        2: coins (1)                                                    [0, 1]
        3: bombs (0.2, 0.4, 0.6, 0.8) + bomb range + explosions (1)     [0, 0.2, 0.4, 0.6, 0.8, 1]
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return np.zeros(params.FEATURE_SHAPE, dtype=float)

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    player_pos = game_state["self"][3]
    others = game_state["others"]

    transformed_state = np.zeros(params.FEATURE_SHAPE)

    # free tiles automatically
    # crates
    result = np.where(field==1)
    for idx in list(zip(result[0], result[1])):
        transformed_state[idx + (0,)] = 0.5
    # stone walls
    result = np.where(field==-1)
    for idx in list(zip(result[0], result[1])):
        transformed_state[idx + (0,)] = 1

    # player position
    transformed_state[player_pos + (1,)] = 1
    # others
    for n, c, b, pos in others:
            transformed_state[pos + (1,)] = 1

    # coins
    for pos in coins:
        transformed_state[pos + (2,)] = 1

    # bombs and explosion map
    transformed_state[:,:,3] = explosion_map
    timer_to_value = { 3: 0.2, 2: 0.4, 1: 0.6, 0: 0.8 }
    for bomb_pos, timer in bombs:
        x,y = bomb_pos
        bomb_value = timer_to_value[timer]
        transformed_state[(x,y,3)] = bomb_value
        wall_x_pos = False
        wall_x_neg = False
        wall_y_pos = False
        wall_y_neg = False
        for i in (1,2,3):
            x_pos, x_neg, y_pos, y_neg = x+i, x-i, y+i, y-i
            if not wall_x_pos:
                if transformed_state[x_pos, y, 0] == 1: # if stone wall
                    wall_x_pos = True
                elif bomb_value > transformed_state[x_pos, y, 3]:
                    transformed_state[(x_pos, y, 3)] = bomb_value
            if not wall_x_neg:
                if transformed_state[x_neg, y, 0] == 1:# if stone wall
                    wall_x_neg = True
                elif bomb_value > transformed_state[x_neg, y, 3]:
                    transformed_state[x_neg, y, 3] = bomb_value
            if not wall_y_pos:
                if transformed_state[x, y_pos, 0] == 1: # if stone wall
                    wall_y_pos = True
                elif bomb_value > transformed_state[x, y_pos, 3]: 
                    transformed_state[x, y_pos,3] = bomb_value
            if not wall_y_neg:
                if transformed_state[x, y_neg, 0] == 1: # if stone wall
                    wall_y_neg = True
                elif bomb_value > transformed_state[x, y_neg, 3]:
                    transformed_state[x, y_neg,3] = bomb_value

    return transformed_state


def extract_box(state, player_pos, vision=3):
    '''
        Extracts box around player of size of vision from first and second dimension of feature matrix
        return value has shape: (2*vision+1, 2*vision+1, ...) e.g. for vision 3 (7, 7, ...)
    '''

    x,y  = player_pos
    x_min = x - vision if x - vision >= 0 else 0
    y_min = y - vision if y - vision >= 0 else 0
    y_max = y_min + 2 * vision + 1
    x_max = x_min + 2 * vision + 1

    if x_max > 17:
        x_max = 17
        x_min = 17 - 2 * vision - 1 
    if y_max > 17:
        y_max = 17
        y_min = 17 - 2 * vision - 1

    return state[x_min:x_max,y_min:y_max,:]