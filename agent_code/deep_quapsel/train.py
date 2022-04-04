from collections import namedtuple, deque

import time
from typing import List
import random
import numpy as np

import events as e
import agent_code.deep_quapsel.dql_params as params
from .features import crate_and_enemy_potential, crate_distance, object_distance, survival_instinct
from .stat_recorder import stat_recorder
from .callbacks import state_to_features, ACTIONS


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
COIN_POS = "COIN_POS"
COIN_NEG = "COIN_NEG"
BOMB_POS = "BOMB_POS"
BOMB_NEG = "BOMB_NEG"
CRATE_NEG = "CRATE_NEG"
CRATE_POS = "CRATE_POS"
BOMB_DESTRUCTIVE ="BOMB_DESTRUCTIVE"
BOMB_NEAR_ENEMY = "BOMB_NEAR_ENEMY"
BOMB_NOT_USEFUL ="BOMD_NOT_USEFUL"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = deque(maxlen=params.REPLAY_MEMORY_SIZE)

    self.timing_train_logger = stat_recorder("./logs/timing_train.log")
    self.reward_logger = stat_recorder("./logs/reward.log")
    self.q_value_logger = stat_recorder("./logs/q_values.log")
    self.defect_logger = stat_recorder("./logs/defect.log")
    self.ep_defect_logger = stat_recorder("./logs/ep_defect.log")

    self.episode_count = 0
    self.update_counter = 0
    self.episode_reward = []
    self.episode_defect = []
    self.ep_rewards = []


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
        coin_dist_old = object_distance(old_field, old_coins, old_player_pos)
        coin_dist_new = object_distance(new_field, new_coins, new_player_pos)
        if len(old_coins) > 0:
            if coin_dist_old > coin_dist_new:
                events.append(COIN_POS)
            elif coin_dist_old < coin_dist_new:
                events.append(COIN_NEG)

        if not e.BOMB_DROPPED in events:
            #### bomb evasion rewards ####
            bomb_old = survival_instinct(old_field, old_bombs, old_explosion_map, old_others, old_player_pos)
            bomb_new = survival_instinct(new_field, new_bombs, new_explosion_map, new_others, new_player_pos)

            # check if agent went to field with lower danger
            if len(old_bombs) > 0:
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


        #### Bomb drop rewards
        if self_action == "BOMB" and not e.INVALID_ACTION in events:
            # reward for dropping bomb near crate or enemy
            crates, enemies = crate_and_enemy_potential(old_field, old_player_pos, old_others)

            self.logger.info(f"Crates: {crates}, enemies: {enemies}")
            if crates > 0 or enemies > 0:
                for i in range(crates):
                    events.append(BOMB_DESTRUCTIVE)
                for i in range(enemies):
                    events.append(BOMB_NEAR_ENEMY)
            else:
                events.append(BOMB_NOT_USEFUL)


        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    current_reward = reward_from_events(self, events)
    self.episode_reward.append(current_reward)

    # Rotate action counterclockwise by self.rotation * 90°
    action_idx = ACTIONS.index(self_action)
    rot_action_idx = (action_idx - self.rotation) % 4 if action_idx < 4 else action_idx
    self_action = ACTIONS[rot_action_idx]

    old_features = state_to_features(old_game_state, self.rotation)
    new_features = state_to_features(new_game_state, self.rotation)
    self.transitions.append(Transition(old_features, self_action, new_features, current_reward))

    if len(self.transitions) < params.MIN_REPLAY_MEMORY_SIZE:
        return

    t1 = time.time()
    # Get a minibatch of random samples from self.transitions
    minibatch = random.sample(self.transitions, params.MINIBATCH_SIZE)

    # Get current features from minibatch, then query NN model for Q values
    current_features = np.array([transition[0] for transition in minibatch])
    current_qs_list = self.model.predict(current_features)

    # Get future states from minibatch, then query NN model for Q values
    new_features = np.array([transition[2] for transition in minibatch])
    future_qs_list = self.target_model.predict(new_features)

    X = []
    y = []

    defect = []
    # Enumerate batches
    for index, (current_state, action, new_current_state, reward) in enumerate(minibatch):

        # Get target Q values
        max_future_q = np.max(future_qs_list[index])
        new_q = reward + params.Q_RATE * max_future_q

        # Update Q value for given state
        current_qs = current_qs_list[index]
        current_q = current_qs[ACTIONS.index(action)]
        current_qs[ACTIONS.index(action)] = new_q

        # Append to our training data
        X.append(current_state)
        y.append(current_qs)

        defect.append(np.square(new_q - current_q))

    defect_acc = np.sum(defect)/len(defect)
    self.defect_logger.write(str(defect_acc))
    self.episode_defect.append(defect_acc)

    # Fit on all samples as one batch
    self.model.fit(np.array(X), np.array(y), batch_size=params.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=None)
    self.timing_train_logger.write(str(round(1000 * (time.time() - t1), 2)))



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
    self.episode_count += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Logs
    self.reward_logger.write(str(np.sum(self.episode_reward) / len(self.episode_reward) if self.episode_reward else 0))
    self.ep_defect_logger.write(str(np.sum(self.episode_defect)/len(self.episode_defect) if self.episode_defect else 0))
    self.episode_reward = []
    self.episode_defect = []

    # Update target network counter every episode
    self.update_counter += 1

    # If counter reaches set value, update target network with weights of main network
    if self.update_counter > params.UPDATE_TARGET:
        self.target_model.set_weights(self.model.get_weights())
        self.update_counter = 0

        # Store the model
        self.model.save_weights(params.MODELNAME)

    # Decay epsilon
    if self.epsilon > params.MIN_EPSILON:
        self.epsilon *= params.EPSILON_DECAY
        self.epsilon = max(params.MIN_EPSILON, self.epsilon)
        self.logger.debug(f"Adjusted epsilon: {self.epsilon}")

    # Save model every 1000 episodes
    if self.episode_count % 1000 == 0:
        self.model.save_weights(f"{params.MODELNAME.split('.')[0]}_{self.episode_count}.h5")

    if self.episode_count == params.STOP_IMITATION_AT:
        self.imitation_rate = 0
    elif self.imitation_rate != 0:
        self.imitation_rate *= params.IMITATION_RATE_DECAY

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 4,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 0.5,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.3,
        COIN_POS: .5,
        COIN_NEG: -.5,
        BOMB_POS: 1,
        BOMB_NEG: -1,
        CRATE_POS: 0.4,
        CRATE_NEG: -0.4,
        BOMB_DESTRUCTIVE: 0.3,
        BOMB_NEAR_ENEMY: 1,
        BOMB_NOT_USEFUL: -1
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
