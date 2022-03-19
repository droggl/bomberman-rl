import os
import pickle
import random
from time import time
from agent_code.deepQuapsel.stat_recorder import stat_recorder

import agent_code.deepQuapsel.dql_params as params

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
        self.logger.info("Loading model from saved state.")
        with open(params.MODELNAME, "rb") as file:
            self.model.set_weights(pickle.load(file))
    self.target_model.set_weights(self.model.get_weights())


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    self.rotation = 0
    # If start of the round: save rotation into self.rotation as self.rotation * 90° counterclockwise
    if params.ROTATION_ENABLED and game_state["step"] == 1:
        starting_pos_to_rotation = {
            (1,1): 0,
            (1,15): 1,
            (15,15): 2,
            (15,1): 3
        }
        self.rotation = starting_pos_to_rotation[game_state["self"][3]]

    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        action_index = np.random.choice(range(6), p=[.2, .2, .2, .2, .1, .1])
    else:
        t1 = time()
        features = state_to_features(game_state, self.rotation)

        if params.TIMING_FEATURES:
            self.timing.write(f"state to features: {round((time()-t1) * 1000,3)}ms")

        t2 = time()
        predictions = self.target_model.predict(np.array(features).reshape(-1, *features.shape))[0]
        self.logger.debug(f"Deciding by argmax from {predictions}")
        action_index = np.argmax(predictions)

        if params.ROTATION_ENABLED: # Rotate action clockwise by 90° * self.rotation
            temp = action_index
            action_index = (action_index + self.rotation) % 4 if action_index < 4 else action_index
            self.logger.info(f"Rotation: {self.rotation}, ACTION: {ACTIONS[temp]}, ROT_ACTION: {ACTIONS[action_index]}")

        if params.TIMING_PREDICT:
            self.timing.write(f"predict: {round((time()-t2) * 1000)}ms")

    return ACTIONS[action_index]


def state_to_features(game_state: dict, rotation) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array of 17x17x8: last dimension maps as follows:
        0: free             [0, 1]
        1: crates           [0, 1]
        2: stone walls      [0, 1]
        3: bombs            [0.25, 0.5, 0.75, 1]
        4: explosion map    [0, 1]
        5: player           [0, 1]
        6: other players    [0, 1]
        7: coins            [0, 1]
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return np.zeros(params.FEATURE_SHAPE, dtype=float)

    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    self = game_state["self"]
    player_pos = self[3]
    others = game_state["others"]

    transformed_state = np.zeros(field.shape + (8,))

    # free tiles
    result = np.where(field==0)
    for idx in list(zip(result[0], result[1])):
        transformed_state[idx + (0,)] = 1

    # crates
    result = np.where(field==1)
    for idx in list(zip(result[0], result[1])):
        transformed_state[idx + (1,)] = 1

    # stone walls
    result = np.where(field==-1)
    for idx in list(zip(result[0], result[1])):
        transformed_state[idx + (2,)] = 1

    # bombs
    timer_to_value = { 3: 0.25, 2: 0.5, 1: 0.75, 0: 1 }
    for bomb_pos, timer in bombs:
        transformed_state[bomb_pos + (3,)] = timer_to_value[timer]

    # explosion map
    transformed_state[:,:,4] = explosion_map

    # player position
    transformed_state[player_pos + (5,)] = 1

    # others
    for n, c, b, pos in others:
        transformed_state[pos + (6,)] = 1

    # coins
    for pos in coins:
        transformed_state[pos + (7,)] = 1

    return np.rot90(transformed_state, rotation)
