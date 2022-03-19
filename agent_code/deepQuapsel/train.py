from collections import namedtuple, deque

import time
import pickle
from typing import List
import agent_code.deepQuapsel.dql_params as params
import random

import numpy as np
from agent_code.deepQuapsel.utils import ModifiedTensorBoard

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify

RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=params.TRANSITION_HISTORY_SIZE)
    self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(params.MODELNAME, int(time.time())))

    self.update_counter = 0



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


    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    if params.ROTATION_ENABLED:
        # Rotate action left by 90° * self.rotation
        action_idx = ACTIONS.index(self_action)        
        rot_action_idx = (action_idx - self.rotation) % 4 if action_idx < 4 else action_idx
        self_action = ACTIONS[rot_action_idx]

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state, self.rotation), self_action, state_to_features(new_game_state, self.rotation), reward_from_events(self, events)))

    if len(self.transitions) < params.MIN_TRANSITIONS_SIZE:
        return

    t1 = time.time()
    # Get a minibatch of random samples from self.transitions
    minibatch = random.sample(self.transitions, params.SMALLBATCH_SIZE)

    # Get current features from minibatch, then query NN model for Q values
    current_features = np.array([transition[0] for transition in minibatch])
    current_qs_list = self.model.predict(current_features)

    # Get future states from minibatch, then query NN model for Q values
    new_features = np.array([transition[2] for transition in minibatch])
    future_qs_list = self.target_model.predict(new_features)

    X = []
    y = []

    # Enumerate batches
    for index, (current_state, action, new_current_state, reward) in enumerate(minibatch):

        # Get new Q from future states
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + params.Q_RATE * max_future_q

        # Update Q value for given state
        current_qs = current_qs_list[index]
        current_qs[ACTIONS.index(action)] = new_q

        # Append to our training data
        X.append(current_state)
        y.append(current_qs)

    # Fit on all samples as one batch
    self.model.fit(np.array(X), np.array(y), batch_size=params.SMALLBATCH_SIZE, verbose=0, shuffle=False, callbacks=None) #[self.tensorboard]) # if terminal_state else None)

    if params.TIMING_TRAIN:
        self.timing.write(f"minibatch training: {round((time.time()-t1) * 1000)}ms")



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
    self.tensorboard.step += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    if params.ROTATION_ENABLED:
        # Rotate action left by 90° * self.rotation
        action_idx = ACTIONS.index(last_action)
        rot_action_idx = (action_idx - self.rotation) % 4 if action_idx < 4 else action_idx
        last_action = ACTIONS[rot_action_idx]

    self.transitions.append(Transition(state_to_features(last_game_state, self.rotation), last_action, state_to_features(None, self.rotation), reward_from_events(self, events)))

    # Update target network counter every episode
    self.update_counter += 1

    # If counter reaches set value, update target network with weights of main network
    if self.update_counter > params.UPDATE_TARGET:
        self.target_model.set_weights(self.model.get_weights())
        self.update_counter = 0

    # Store the model
    with open(params.MODELNAME, "wb") as file:
        pickle.dump(self.model.get_weights(), file)


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
        e.WAITED: -0.3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum