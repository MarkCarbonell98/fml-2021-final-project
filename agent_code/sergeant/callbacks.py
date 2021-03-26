import json
import numpy as np
import events as e
import settings as s
import sys
import os
sys.path.append('agent_code/sergeant/')
from train import *
from utils import *

INVALID_ACTION = 6
DROP_BOMB = 4
KILLED_SELF = 13
GOT_KILLED = 14

def setup(self):
    self.curr_state = None
    self.prev_state = None
    self.episode_rewards_log = dict()

    # ACTIONS AND EVENTS
    self.actions = ['LEFT', 'UP', 'RIGHT', 'DOWN', 'BOMB', 'WAIT']  # Actions in the needed order.
    self.changes_point_in_dir = {'left': [-1, 0], 'up': [0, -1], 'right': [1, 0],
                                 'down': [0, 1], 0: [-1, 0], 1: [0, -1], 2: [1, 0], 3: [0, 1]}

    self.relevant_events = [0, 1, 2, 3, 4, 6, 7, 14]
    self.events = {0: 0, 1: 2, 2: 1, 3: 3, 7: 4, 4: 5, 6: 6, 14: 14}

    self.rewards = {'bomb': -70, 'danger1': -40, 'coin': 80, 'priority': 60, 'enemy': -10,
                    'wall': -70, 'empty': -10, 'danger2': -30, 'danger3': -20, 'invalid': -70, 'goal_action': 80,
                    'dead': -70, 'danger': -40, 'crate': -70}

    # HYPERPARAMETERS
    self.gamma = 0.65
    self.alpha = 0.25
    # Epsilon-greedy policy
    self.epsilon = 1.0
    self.epsilon_min = 0.20  # Relatively high minimum eps would prevent overfitting
    self.epsilon_decay = 0.9995

    # self.radius permits to look for priorities nearby. self.weights establish priorities for the agent
    self.radius = None
    self.radius_incrementer = 0.1  # Each turn, radius is incremented by 0.05
    self.weights = {'field': 1, 'field': 30, 'others': 1}  # Weights for each object on the board.

    # Load Q_table:
    try:  # If q_table.json file doesn't exit
        read_dict_from_file(self)
    except Exception as ex:
        self.q_table = dict()

def act(self, game_state):
    """

    Args:
        self:

    Returns: str is an action
        

    """
    self.game_state = game_state
    self.logger.info('Pick action according to pressed key')
    self.game_state['crate_density'] = s.CRATE_DENSITY

    # Epsilon-greedy policy
    # game_events['crate_density'] # error 
    if self.game_state['step'] == 1 and self.radius is None:
        if s.CRATE_DENSITY == 0:
            training_radius(self)
        else:
            real_radius(self)

    # First step: find the current state.
    self.curr_state = find_state(self)
    # Second step:
    string = state_to_str(self.curr_state)

    if self.train:
        action_rewards = np.array(self.q_table[string])
        if 0 in action_rewards:

            indices = np.where(action_rewards == 0)[0]
            action = np.random.choice(indices)


        # Epsilon-Greedy Policy: Epsilon is updated after every finished round while training.
        # Exploration- Exploitation Policy: epsilon_min permits the agent to still try random actions for different states
        elif np.random.uniform(0, 1) < self.epsilon:
            # Speed-up training: agent is encouraged to do an action for a state that has no action info
            action = np.random.randint(0, 6)
        else:
            action = np.argmax(self.q_table[string])

        # Prepared q_table should be used when not training
    else:
        try:
            # Exception for not known states. A random action is chosen
            knowledge_list = np.array(self.q_table[state_to_str(self.curr_state)])
            best_indices = np.where(knowledge_list == max(knowledge_list))[0]
            action = np.random.choice(best_indices)
        except KeyError as ke:
            action = np.random.randint(0, 6)

    # Set action and change radius.
    self.radius += self.radius_incrementer
    self.next_action = self.actions[action]
    self.logger.info(self.actions[action])
    return self.actions[action]




