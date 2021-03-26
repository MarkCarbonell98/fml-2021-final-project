import json
import numpy as np
import events as e
import settings as s
import sys
import os

INVALID_ACTION = 6
DROP_BOMB = 4
KILLED_SELF = 13
GOT_KILLED = 14

def update_reward(self):
    """
    Allow intermediate rewards based on game events.

    Args:
        self:

    Returns:

    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    # Restart round log
    if self.game_state['step'] == 1:
        self.round_rewards_log = dict()
        self.train_flag = True
        training_radius(self) if s.crate_density == 0 else real_radius(self)

    # No action has been made.
    if len(self.events) == 0:
        return

    self.curr_pos = np.array([self.game_state["self"][0], self.game_state["self"][1]])
    state_string = state_to_str(self.curr_state)
    # First step: We created a list of relevant actions from given events
    action, reward_update = 0, 0
    for event in self.events:
        # Move has been made:
        if event not in self.relevant_actions:
            continue

        # Translation of game settings
        action = self.actions.index(self.next_action)

        if event == GOT_KILLED:
            reward_update += self.rewards['dead']

        if event == INVALID_ACTION:
            action = self.actions.index(self.next_action)
            if self.curr_state[action] == 'priority':
                reward_update += self.rewards[self.curr_state[action]]
            else:
                reward_update += self.rewards['invalid']

        if action < DROP_BOMB:
            curr_action = self.curr_state[action]
            reward_update += self.rewards[curr_action]

        elif action == DROP_BOMB:  # action == 4
            if 'crate' in self.curr_state[:4] or 'enemy' in self.curr_state[:4]:
                reward_update += self.rewards['goal_action']

        else:
            reward_update += -5 if self.curr_state[4] != 'danger' else self.rewards['bomb']

        try:
            # Implementation Q-Learning algorithm
            prev_state_string = state_to_str(self.prev_state)
            alpha, gamma = self.alpha, self.gamma
            Q0sa0 = self.q_table[prev_state_string][self.prev_action]
            Qs1a1 = self.rewards['dead'] if GOT_KILLED == self.events[-1] else max(self.q_table[state_string])
            curr_reward = Q0sa0 + alpha * (self.prev_reward + gamma * Qs1a1 - Q0sa0)
            self.q_table[prev_state_string][self.prev_action] = curr_reward

        except Exception as ex:
            # No previous steps are listed
            pass

        self.prev_action = action
        self.prev_reward = reward_update
        self.prev_state = self.curr_state






def end_of_round(self):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.
    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.
    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    # Update reward
    update_reward(self)
    update_reward(self)

    # Reduce epsilon
    if self.epsilon >= self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    # For each round, the Q function variables have to be restarted
    self.curr_state = None
    self.prev_state = None
    self.prev_action = None

    # Save q_table
    write_dict_to_file(self)

    if self.number_of_games % 50 == 0:
        print("#games = {}".format(self.number_of_games))
