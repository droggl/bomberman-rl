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

from .features import *
from .online_gradient_boosting import online_gradient_boost_regressor as ogbr
import agent_code.agent1.train_params as tparam
from .stat_recorder import stat_recorder

# This is only an example!
Step = namedtuple('Step',
                    ('state', 'action', 'reward'))
N_Transition = namedtuple('Transition',
                        ('state', 'action', 'final_state', 'acc_reward'))

# Events
COIN_POS = "COIN_POS"
COIN_NEG = "COIN_NEG"
BOMB_POS = "BOMB_POS"
BOMB_NEG = "BOMB_NEG"
CRATE_NEG = "CRATE_NEG"
CRATE_POS = "CRATE_POS"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Setup an array that will note transition tuples
    self.transitions = deque(maxlen=tparam.TRANSITION_HISTORY_SIZE)

    # Init buffer for n-step Q-Learning
    self.transition_buffer = deque(maxlen=tparam.Q_STEPS)

    self.rho = tparam.RHO_START
    self.episode_number = 0

    # set up new model
    if tparam.RESET == True or not os.path.isfile(tparam.MODEL_NAME):
        self.logger.info("Setting up model \"{}\" from scratch.".format(tparam.MODEL_NAME))

        # init current model
        # TODO evaluate init
        self.model_current = []
        for i in range(6):
            self.model_current.append(ogbr(GBR, tparam.GB_RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE))
    # load existing model
    else:
        self.logger.info("Loading model \"{}\" from saved state.".format(tparam.MODEL_NAME))
        with open(tparam.MODEL_NAME, "rb") as file:
            self.model_current = pickle.load(file)

    # init new(replacement) model
    self.model_new = []
    for i in range(6):
        self.model_new.append(ogbr(GBR, tparam.GB_RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE))


    self.defect = stat_recorder("./logs/defect.log", tparam.RESET)


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

    #### Rewards ####

    if old_game_state is not None and new_game_state is not None:

        old_field = old_game_state["field"]
        old_bombs = old_game_state["bombs"]
        old_explosion_map = old_game_state["explosion_map"]
        old_coins = old_game_state["coins"]
        old_player_pos = old_game_state["self"][3]
        old_others = old_game_state["others"]

        new_field = new_game_state["field"]
        new_bombs = new_game_state["bombs"]
        new_explosion_map = new_game_state["explosion_map"]
        new_coins = new_game_state["coins"]
        new_player_pos = new_game_state["self"][3]
        new_others = new_game_state["others"]
        
        #### Coin finder rewards ####
        coin_dist_old = coin_distance(old_field, new_coins, old_player_pos)
        coin_dist_new = coin_distance(new_field, new_coins, new_player_pos)

        if coin_dist_old > coin_dist_new:
            events.append(COIN_POS)
        elif coin_dist_old < coin_dist_new:
            events.append(COIN_NEG)

        #### bomb evasion rewards ####
        bomb_old = survival_instinct(old_field, old_bombs, old_explosion_map, old_others, old_player_pos)
        bomb_new = survival_instinct(new_field, new_bombs, new_explosion_map, new_others, new_player_pos)

        # check if agent went to field with lower danger
        if bomb_new[4] < bomb_old[4]:
            events.append(BOMB_POS)
        if bomb_new[4] > bomb_old[4]:
            events.append(BOMB_NEG)


        #### Crate finder rewards ####
        crate_dist_old = crate_distance(old_field, old_player_pos)
        crate_dist_new = crate_distance(new_field, new_player_pos)

        if crate_dist_old > crate_dist_new:
            events.append(CRATE_POS)
        elif crate_dist_old < crate_dist_new:
            events.append(CRATE_NEG)

    #### N-step Q-Learning ####

    old_features = state_to_features(old_game_state)

    # if state buffer is full
    if len(self.transition_buffer) == tparam.Q_STEPS:
        # get oldest from queue
        (state, action, reward) = self.transition_buffer.popleft()

        # accumulate reward
        gamma = tparam.Q_RATE
        for (f_state, f_action, f_reward) in self.transition_buffer:
            reward = reward + gamma * f_reward
            gamma = tparam.Q_RATE * gamma

        # add to transitions (training set)
        self.transitions.append(N_Transition(state, action, old_features, reward))

    # add new state to buffer
    self.transition_buffer.append(Step(old_features, self_action, reward_from_events(self, events)))



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

    t_0 = time.time() # timing
 
    #### empty transition buffer ####

    last_features = state_to_features(last_game_state)

    # add transition for last state-action
    # NOTE: Out of order
    self.transitions.append(N_Transition(last_features, last_action, None, reward_from_events(self, events)))

    while len(self.transition_buffer) > 0:
        # get oldest from queue
        (state, action, reward) = self.transition_buffer.popleft()

        # accumulate reward
        gamma = tparam.Q_RATE
        for (f_state, f_action, f_reward) in self.transition_buffer:
            reward = reward + gamma * f_reward
            gamma = tparam.Q_RATE * gamma

        # add to transitions (training set)
        self.transitions.append(N_Transition(state, action, last_features, reward))
        last_features = None


    #### prepare training data ####

    # number of buffered transitions
    n = len(self.transitions)

    # wait for buffer to fill
    if tparam.BUFFER_CLEAR and n < tparam.TRANSITION_HISTORY_SIZE:
        return

    # transfer transition queue to numpy

    state_arr = np.empty((n,tparam.FEATURE_LEN))
    action_arr = np.empty(n, dtype=np.int8)
    final_arr = np.empty((n, tparam.FEATURE_LEN))
    reward_arr = np.empty(n)

    missing_next = []
    missing_last = []

    for (i, (state, action, final, reward)) in enumerate(self.transitions):
        # missing next state treatment
        if final is None:
            # placeholder next
            final_arr[i] = np.zeros(tparam.FEATURE_LEN)
            # remember
            missing_next.append(i)
        else:
            final_arr[i] = final

        # missing last state treatment
        if state is None:
            state_arr[i] = np.zeros(tparam.FEATURE_LEN)
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
        
        state_arr[i] = state
        reward_arr[i] = reward

    # evaluate current estimators for Q(final)
    Q_pred = []
    for i in range(6):
        Q_pred.append(self.model_current[i].predict(final_arr))

    # compute expectation Y
    Q_max = np.amax(np.stack(Q_pred), axis = 0)

    # missing final -> no expected future reward
    for i in missing_next:
        Q_max[i] = 0

    # missing state -> ignore
    # TODO better idea?
    for i in missing_last:
        action_arr[i] = 6

    # compute expectation Y
    Y = reward_arr + tparam.Q_RATE**tparam.Q_STEPS * Q_max


    #### model performance logging
    acc = 0
    count = 0
    for i in range(6):
        mask = (action_arr == i)
        count = count + sum(mask)
        if sum(mask) > 0:
            defect = Y[mask] - self.model_current[i].predict(state_arr[mask])
            acc = acc + np.sum(np.square(defect))
    self.defect.write(str(acc / count))


    #### Update Q estimators ####

    # update Q estimators 
    for i in range(6):
        mask = (action_arr == i)
        if sum(mask) > 0:
            self.model_new[i].fit_update(state_arr[mask], Y[mask])

    # replace current model with new after CYCLE_TIME iterations
    self.episode_number = self.episode_number + 1

    if self.episode_number % tparam.CYCLE_TIME == 0:
        self.logger.info("Replacing model.")

        # replace current with new and fresh init new
        for i in range(6):
            self.model_current[i] = self.model_new[i]
            self.model_new[i] = ogbr(GBR, tparam.GB_RATE, n_estimators=tparam.WEAK_N_EST, learning_rate=tparam.WEAK_RATE)

    # timing
    t_1 = time.time()
    self.logger.info("Model update (ms): {}".format((t_1-t_0) * 1000))

    # clear transition buffer
    if tparam.BUFFER_CLEAR:
        self.transitions.clear()

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
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 0.5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        COIN_POS: .5,
        COIN_NEG: -.5,
        BOMB_POS: .3,
        BOMB_NEG: -.3,
        CRATE_POS: 0.1,
        CRATE_NEG: -0.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum