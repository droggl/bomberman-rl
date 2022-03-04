from collections import namedtuple, deque
from xmlrpc.client import Boolean

import numpy as np
import pickle
import time
import os
from typing import List

from sklearn.ensemble import GradientBoostingRegressor as GBR

import events as e
from .callbacks import state_to_features, ACTIONS

from .aux import coin_collector2, survival_instinct
from .online_gradient_boosting import online_gradient_boost_regressor as ogbr
import agent_code.agent1.train_params as tparam

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
COIN_POS = "COIN_POS"
COIN_NEG = "COIN_NEG"
BOMB_POS = "BOMB_POS"
BOMB_NEG = "BOMB_NEG"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=tparam.TRANSITION_HISTORY_SIZE)
    self.rho = tparam.RHO_START
    self.episode_number = 0

    # set up new model
    if tparam.RESET == True or not os.path.isfile(tparam.MODEL_NAME):
        self.logger.info("Setting up model \"{}\" from scratch.".format(tparam.MODEL_NAME))

        # init current model
        # TODO evaluate init
        self.model_current = []
        for i in range(6):
            self.model_current.append(ogbr(GBR, tparam.RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE))
    # load existing model
    else:
        self.logger.info("Loading model \"{}\" from saved state.".format(tparam.MODEL_NAME))
        with open(tparam.MODEL_NAME, "rb") as file:
            self.model_current = pickle.load(file)

    # init new model
    self.model_new = []
    for i in range(6):
        self.model_new.append(ogbr(GBR, tparam.RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None and new_game_state is not None:
        
        #### Coin finder rewards ####
        coin_old = coin_collector2(old_game_state["field"], old_game_state["coins"], old_game_state["self"][3])
        coin_new = coin_collector2(new_game_state["field"], new_game_state["coins"], new_game_state["self"][3])

        old_max = np.max(coin_old[0:4])
        new_max = np.max(coin_new[0:4])
        # print(old_max, new_max)

        # check if agent moved closer to coin
        if new_max > old_max:
            events.append(COIN_POS)
        elif new_max < old_max:
            events.append(COIN_NEG)

        #### bomb evasion rewards ####
        bomb_old = survival_instinct(old_game_state["field"], old_game_state["bombs"], old_game_state["explosion_map"], old_game_state["self"][3])
        bomb_new = survival_instinct(new_game_state["field"], new_game_state["bombs"], new_game_state["explosion_map"], new_game_state["self"][3])

        # check if agent went to field with lower danger
        if bomb_new[4] < bomb_old[4]:
            events.append(BOMB_POS)
        elif bomb_new[4] > bomb_old[4]:
            events.append(BOMB_NEG)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    last_features = state_to_features(last_game_state)
    D = len(last_features)
    self.transitions.append(Transition(last_features, last_action, None, reward_from_events(self, events)))

    t_0 = time.time() # timing

    # number of buffered transitions
    n = len(self.transitions)

    # transfer transition queue to numpy

    last_arr = np.empty((n,D))
    action_arr = np.empty(n, dtype=np.int8)
    next_arr = np.empty((n, D))
    reward_arr = np.empty(n)

    missing_next = []
    missing_last = []

    for (i, (last, action, next, reward)) in enumerate(self.transitions):
        # missing next state treatment
        if next is None:
            # placeholder next
            next_arr[i] = np.zeros(tparam.FEATURE_LEN)
            # remember
            missing_next.append(i)
        else:
            next_arr[i] = next

        # missing last state treatment
        if last is None:
            last_arr[i] = np.zeros(tparam.FEATURE_LEN)
            missing_last.append(i)

        # translate action to integer enumeration
        if action == "UP":
            action_arr[i] = 0
        elif action == "RIGHT":
            action_arr[i] = 1
        elif action == "DOWN":
            action_arr[i] = 2
        elif action == "LEFT":
            action_arr[i] = 3
        elif action == "WAIT":
            action_arr[i] = 4
        elif action == "BOMB":
            action_arr[i] = 5
        
        last_arr[i] = last
        reward_arr[i] = reward

    # evaluate current estimators for Q
    Q_pred = []
    for i in range(6):
        Q_pred.append(self.model_current[i].predict(next_arr))

    # compute expectation Y
    Q_max = np.amax(np.stack(Q_pred), axis = 0)

    # missing next -> no expected future reward
    for i in missing_next:
        Q_max[i] = 0

    # missing last -> ignore
    # TODO better idea?
    for i in missing_last:
        action_arr[i] = 6

    # compute expectation Y
    Y = reward_arr + tparam.GAMMA * Q_max

    # update Q estimators 
    for i in range(6):
        mask = (action_arr == i)
        if sum(mask) > 0:
            self.model_new[i].fit_update(last_arr[mask], Y[mask])

    # replace current model with new after CYCLE_TIME iterations
    self.episode_number = self.episode_number + 1

    if self.episode_number % tparam.CYCLE_TIME == 0:
        self.logger.info("Replacing model.")

        # replace current with new and fresh init new
        for i in range(6):
            self.model_current[i] = self.model_new[i]
            self.model_new[i] = ogbr(GBR, tparam.RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE)

    # timing
    t_1 = time.time()
    self.logger.info("Model update (ms): {}".format((t_1-t_0) * 1000))

    # additional logging
    # with open("./logs/round_length", "a") as file:
        # file.write(str(last_game_state["step"]) + " ")

    # Store the model
    with open(tparam.MODEL_NAME, "wb") as file:
        pickle.dump(self.model_current, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 0.5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        COIN_POS: .1,
        COIN_NEG: -.1,
        BOMB_POS: .5,
        BOMB_NEG: -.5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
