import pickle
import os
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .callbacks import features_to_state_number
from .callbacks import state_finder
from scipy.sparse import dok_matrix, coo_matrix
from scipy import sparse

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 8 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXTRA_REWARD = 0
REWARD_DISTANCE = 15
# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

REPETION_SEARCH = 5


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.repition = deque(maxlen=3)
    #self.samefeature = deque(maxlen=REPETION_SEARCH)
    self.bomb_dropped = deque(maxlen= 1)
    self.states = np.array([])
    self.coins_collected = REPETION_SEARCH
    max_distance = 3  # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES, TAMBIEN EN CALLBACKS
    self.wrong_moves = deque(maxlen=1)
    possible_outcomes = max_distance * 2 + 1
    number_of_space_taker = 3
    self.steps_away_from_bomb = deque(maxlen=4)
    row = np.array([])
    col = np.array([])
    data = np.array([])
    self.good_bomb = False
    r = 4
    q_values = np.zeros((1000000, 6))
    # q_values = np.ones((201411,6))
    if self.train and os.path.isfile("q_values.npy"):
        q_values = np.load("q_values.npy")
        print("File was loaded")
    self.q_values = q_values

def action_symmetry_up_down(action): #ESTO FUE CAMBIADO
    if action == 0:
        return 2
    if action == 2:
        return 0
    else:
        return action
def symmetry_up_down(features):
    next_to_crate, free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate = features
    direction_of_nearest_crate = action_symmetry_up_down(direction_of_nearest_crate) #ESTO FUE AGREGADO
    #if m1y==5:
        #m1y=-5
    free_spaces_complete = np.flipud(free_spaces_complete.T).T
    return (next_to_crate,free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate)

def action_symmetry_left_right(action):
    if action ==1:
        return 3
    if action ==3:
        return 1
    else:
        return action

def symmetry_left_right(features):
    next_to_crate, free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate= features
    #if m1x==5:
        #m1x=-5
    direction_of_nearest_crate = action_symmetry_left_right(direction_of_nearest_crate)
    free_spaces_complete = np.fliplr(free_spaces_complete.T).T

    return (next_to_crate,free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate)

def action_symmetry_x_y(action):
    if action ==2:
        return 1
    if action ==1:
        return 2
    if action ==0:
        return 3
    if action ==3:
        return 0
    else:
        return action

def symmetry_x_y(features):
    next_to_crate, free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate = features
    """if x_better ==0:
        x_better =1
    if x_better ==1:
        x_better = 0"""
    direction_of_nearest_crate = action_symmetry_x_y(direction_of_nearest_crate)
    free_spaces_complete = np.rot90(np.fliplr(free_spaces_complete.T)).T
    return (next_to_crate,free_spaces_complete, bomb_timer1,danger,direction_of_nearest_crate)


def use_symmetry(features, action, next_features, reward, self):
    if features is None or next_features is None:
        return
    old_value = self.q_values[features_to_state_number(features), action]
    #print("Old Value: " + str(old_value) +str(type(old_value)))
    #print(type(np.float64(0.3)))
    #print(type(old_value) == type(np.float64(0.3)))
    #print('Hola')
    #print(action)
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(next_features)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        self.q_values[features_to_state_number(features), action] = new_value
        if old_value == 0:
            self.logger.info("Agent found a new state: 0")

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    #(d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    #print(action_changed)
    old_value = self.q_values[state_number_changed, action_changed ]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 1")
    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    #print(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    #print(action_changed)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 2")

    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    #print(action_changed)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 3")

    # Symmetry 4, interchange of x and y:
    features = symmetry_x_y(features) #Ultima vez que usamos todos estos valores, entonces podemos cambiarlo por facilidad
    next_features = symmetry_x_y(next_features)
    state_number_changed = features_to_state_number(features)
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action = action_symmetry_x_y(action)
    #print(action)
    old_value = self.q_values[state_number_changed, action]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(next_features)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 4")
    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 4.1")
    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 4.2")
    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(
            self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value
        if old_value ==0:
            self.logger.info("Agent found a new state: 4.3")

def action_number(action):
    if action == "UP":
        action = 0
    if action == "RIGHT":
        action = 1
    if action == "DOWN":
        action = 2
    if action == "LEFT":
        action = 3
    if action == "WAIT":
        action = 4
    if action == "BOMB":
        action = 5
    return action


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
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
    if self_action is not None:
        # self.states  = np.append(state_finder(old_game_state))
        # if len(self.states)==5:
        # values, counts = np.unique(self.states, return_counts=True)
        # self.samefeature.append()
        #print(reward_from_events(self, events))
        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
        self.repition.append(state_finder(old_game_state))
        repition = True
        if state_finder(old_game_state) == state_finder(new_game_state):
            self.logger.info("Agent waited without reason")
            features = state_to_features(old_game_state)
            next_features = state_to_features(new_game_state)
            action = self_action
            reward = -30
            action = action_number(action)
            use_symmetry(features, action, next_features, reward, self)

        if self.good_bomb:
            self.steps_away_from_bomb.append(
                Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                           reward_from_events(self, events)))
        #print(symmetry_x_y(state_to_features(old_game_state)))
        #print(symmetry_up_down(state_to_features(old_game_state)))
        #print(symmetry_left_right(state_to_features(old_game_state)))

        #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
        # print(events)
        features = state_to_features(old_game_state)
        #print(features)
        #print(symmetry_up_down(features))
        #print(symmetry_left_right(features))
        #print(symmetry_x_y(features))
        #next_features = state_to_features(new_game_state)
        #next_state = features_to_state_number(next_features)
        #state = features_to_state_number(features)


        next_to_crate, free_spaces_complete, bomb_timer1, danger,direction_of_nearest_crate = state_to_features(old_game_state)
        xm = 4
        ym = 4
        xp = 4
        yp = 4
        xmn = 0
        ymn = 0
        xpn = 0
        ypn = 0
        for i in range(4):
            if free_spaces_complete.T[4, 3 - i] == 0:
                xm = i
                break
        for i in range(4):
            if free_spaces_complete.T[3 - i, 4] == 0:
                ym = i
                break
        for i in range(4):
            if free_spaces_complete.T[4, 5 + i] == 0:
                xp = i
                break
        for i in range(4):
            if free_spaces_complete.T[5 + i, 4] == 0:
                yp = i
                break
        #print(free_spaces_complete.T)
        free_spaces_complete.T[4 + 1, 4] = 0
        free_spaces_complete.T[4 - 1, 4] = 0
        free_spaces_complete.T[4, 4 + 1] = 0
        free_spaces_complete.T[4, 4 - 1] = 0
        for i in range(xm):
            i = i + 1
            if free_spaces_complete.T[4 + 1, 4 - i] == 1 or free_spaces_complete.T[4 - 1, 4 - i] == 1:
                xmn = 1
                break
        for i in range(ym):
            i = i + 1
            if free_spaces_complete.T[4 - i, 4 + 1] == 1 or free_spaces_complete.T[4 - i, 4 - 1] == 1:
                ymn = 1
                break
        for i in range(xp):
            i = i + 1
            if free_spaces_complete.T[4 + 1, 4 + i] == 1 or free_spaces_complete.T[4 - 1, 4 + i] == 1:
                xpn = 1
                break
        for i in range(yp):
            i = i + 1
            if free_spaces_complete.T[4 + i, 4 + 1] == 1 or free_spaces_complete.T[4 + i, 4 - 1] == 1:
                ypn = 1
                break
        #print(ym, yp, xm, xp, ymn, ypn, xmn, xpn)
        self.wrong_moves.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                           reward_from_events(self, events)))

        if xp==0:
            if self_action =='RIGHT':
                self.logger.info("Wrong move")
                features, action, next_features, reward = self.wrong_moves.pop()
                reward = reward - 3
                action = self_action
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
        if xm==0:
            if self_action =='LEFT':
                self.logger.info("Wrong move")
                features, action, next_features, reward = self.wrong_moves.pop()
                reward = reward - 3
                action = self_action
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
        if yp==0:
            if self_action =='DOWN':
                self.logger.info("Wrong move")
                features, action, next_features, reward = self.wrong_moves.pop()
                reward = reward - 3
                action = self_action
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
        if ym==0:
            if self_action =='UP':
                self.logger.info("Wrong move")
                features, action, next_features, reward = self.wrong_moves.pop()
                reward = reward - 3
                action = self_action
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
        if danger==1:
            if self_action == 'WAIT':
                self.logger.info("In danger, and waited")
                features, action, next_features, reward = self.wrong_moves.pop()
                reward = reward - 3
                action = self_action
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)



        numpy_events = np.array(events)
        if np.any(np.isin(numpy_events, 'BOMB_DROPPED')):
            self.bomb_dropped.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))

            #print(ym, yp, xm, xp, ymn, ypn, xmn, xpn)
            if xmn ==0 and ymn ==0 and xpn ==0 and ypn ==0 and xm<4 and xp<4 and ym <4 and yp<4:
                self.logger.info("The Agent dropped a bomb when he should not")
                self.good_bomb = False
                features, action, next_features, reward = self.bomb_dropped.pop()
                reward = reward - 50
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
            if not(xmn ==0 and ymn ==0 and xpn ==0 and ypn ==0 and xm<4 and xp<4 and ym <4 and yp<4):
                self.logger.info("The Agent dropped a bomb correctly")
                self.good_bomb = True
                bomb = self.bomb_dropped.pop()
                features, action, next_features, reward = bomb
                reward = reward + 1
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
                self.bomb_dropped.append(bomb)
        if np.any(np.isin(numpy_events, 'CRATE_DESTROYED')) and self.bomb_dropped:
            number_of_crates = np.sum(np.where(numpy_events== 'CRATE_DESTROYED', 1,0))
            number_of_coins = np.sum(np.where(numpy_events == 'COIN_FOUND', 1, 0))
            features, action, next_features, reward = self.bomb_dropped.pop()
            reward = reward + number_of_crates* 0.5 + number_of_coins*0.2
            action = action_number(action)
            use_symmetry(features, action, next_features, reward, self)
        if np.any(np.isin(numpy_events, 'BOMB_EXPLODED')) and not np.any(np.isin(numpy_events, 'KILLED_SELF')):
            self.good_bomb = False
            self.logger.info("Damn yo, good steps!")
            if not np.any(np.isin(numpy_events, 'CRATE_DESTROYED')) and self.bomb_dropped:
                self.logger.info("Drop a bomb when there were no crates near")
                features, action, next_features, reward = self.bomb_dropped.pop()
                reward = reward - 3
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
            while self.steps_away_from_bomb:
                features, action, next_features, reward = self.steps_away_from_bomb.pop()
                reward = reward + 2
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)

            """if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            print("coin was collected")
            self.coins_collected = self.coins_collected + 1
            self.steps = 0"""

        """## AQUI ESTABA EL CODIGO PARA LA DISTANCIA
        coin_colected = False
        if next_state is not None and state is not None:
            next_max = np.max(self.q_values[next_state])
            action = self_action
            action = action_number(action)
            #old_value = self.q_values[state, action]
            if (distance_old<distance_new) and not np.any(np.isin(numpy_events, 'COIN_COLLECTED')): #or (distance_old==distance_new) ; and distance_old>3
                reward = -REWARD_DISTANCE
                #new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                use_symmetry(features, action, next_features, reward, self)
                print("Agent got farther of coin")
            if distance_old>distance_new and not np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
                reward = REWARD_DISTANCE
                #new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                use_symmetry(features, action, next_features, reward, self)
                print("Agent got closer of coin")"""

        if np.any(np.isin(numpy_events, 'INVALID_ACTION')) and self.transitions:
            #if len(self.repition) == 1:
                #repition = False
            #self.repition.pop()
            last_transition = self.transitions.pop()
            self.logger.info("Agent did an invalid move")
            features, action, next_features, reward = last_transition
            next_state = features_to_state_number(next_features)
            state = features_to_state_number(features)
            reward = reward - 50
            if features is not None and next_features is not None and action is not None:
                action = action_number(action)
                use_symmetry(features, action, next_features, reward, self)
                """next_max = np.max(self.q_values[next_state])
                old_value = self.q_values[state, action]
                if type(old_value)== type(np.float64(0.3)):
                    new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                    u, d, l, r, m1x, m1y, n, near, distance, state = features

                    use_symmetry(u, d, l, r, m1x, m1y, n, near, distance, state, action, new_value, self)"""

                # print("Agent was punished: INVALID MOVE")
                # np.save("q_values.npy", self.q_values)
                # print("An invalid move was made and the file was saved")
                # self.steps = self.steps - 1

        # print("Not empty")
        # print("Length of repition deque: " + str(len(self.repition)))
        # print("Length of transition deque: " + str(len(self.transitions)))
        """if repition:
            if len(self.repition) >=2:
                if np.all(self.repition[0] == self.repition[1]):
                    #print("Repetion was made #1")
                    while self.repition:
                        features, action, next_features, reward = self.transitions.pop()
                        reward = reward - 0.5
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                        self.repition.pop()
                        self.steps = self.steps - 1
            if len(self.repition) == 5:
                if np.all(self.repition[0] == self.repition[5]):
                    #print("Repetion was made #2")
                    while self.repition:
                        features, action, next_features, reward = self.transitions.pop()
                        reward = reward - 0.5
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                        self.repition.pop()
                        self.steps = self.steps - 1

                self.repition.clear()
                # repition = False
                #print("Repition was cleared")
        """
        # state_to_features is defined in callbacks.py
        if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            coin_colected = True
            while self.transitions:
                features, action, next_features, reward = self.transitions.pop()
                next_state = features_to_state_number(next_features)
                state = features_to_state_number(features)
                if features is not None and next_features is not None and action is not None:
                    action = action_number(action)
                    use_symmetry(features, action, next_features, reward, self)
                    # next_features = features
                    # features = old_features
                    # before_features, action, features, reward = self.transitions.pop()
                    # action = action_number(action)
                    # use_symmetry(features, action, next_features, reward, self)
                    # next_features = features
                    # features = before_features
                    # use_symmetry(features, action, next_features, reward, self)

            self.transitions.clear()
            self.repition.clear()
            # np.save("q_values.npy", self.q_values)
            # print("1 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0
        if self.steps >= TRANSITION_HISTORY_SIZE and self.transitions:
            features, action, next_features, reward = self.transitions.pop()
            next_state = features_to_state_number(next_features)
            state = features_to_state_number(features)
            if state is not None and next_state is not None and action is not None:
                while self.transitions:
                    if state is not None and next_state is not None and action is not None:
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                        features, action, next_features, reward = self.transitions.pop()
                        next_state = features_to_state_number(next_features)
                        state = features_to_state_number(features)
            self.transitions.clear()
            self.repition.clear()
            #print("0 Coins were collected and Q-Table got updated")
            self.coins_collected = 0
            self.steps = 0
    #if self_action == None:
        #print("Self_action = None")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.info("A round ended")
    numpy_events = np.array(events)
    if np.any(np.isin(numpy_events, 'KILLED_SELF')) and self.good_bomb:
        self.logger.info("If going to kill yourself: \n    return don't")
        while self.steps_away_from_bomb:
            features, action, next_features, reward = self.steps_away_from_bomb.pop()
            reward = reward -0.2
            action = action_number(action)
            use_symmetry(features, action, next_features, reward, self)


    self.rounds = self.rounds + 1
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_finder(last_game_state), last_action, None, reward_from_events(self, events)))
    end_features = state_to_features(last_game_state)
    # u_end, d_end, l_end, r_end, m1x_end, m1y_end, n_end, near_end, distance_end, state_end = end_features
    end_state = features_to_state_number(end_features)
    action = last_action
    action = action_number(action)
    end_action = action
    end_reward = reward_from_events(self, events)
    #print(end_reward)
    if (end_features is not None and end_action is not None) and self.transitions:
        features, action, vorletzter_features, reward = self.transitions.pop()
        if (end_state is not None and end_action is not None and vorletzter_features is not None):
            use_symmetry(vorletzter_features, end_action, end_features, end_reward, self)
        state = features_to_state_number(features)
        action = action_number(action)
        if vorletzter_features is not None and action is not None:
            use_symmetry(features, action, vorletzter_features, reward, self)
        while self.transitions:
            features, action, next_features, reward = self.transitions.pop()
            action = action_number(action)
            use_symmetry(features, action, next_features, reward, self)
    self.transitions.clear()
    #print("deque was cleared")
    #print("Game Ended")
    self.repition.clear()
    if self.rounds%1000 ==0:
        self.logger.info('File was saved')
        np.save("q_values.npy", self.q_values)
        self.ruled_based_agent = not self.ruled_based_agent
        self.move = not self.move
        if self.ruled_based_agent:
            self.logger.info('Ruled Based agent is playing')
        else:
            self.logger.info('The Crate Destroyer is playing')
    if self.rounds % 2500 == 0:
        self.random_prob = self.random_prob * 0.9
        self.logger.info('Random_prob: '+ str(self.random_prob))

    if self.rounds % 10000==0:
        self.alpha = self.alpha * 0.999
        self.gamma = self.gamma * 0.999
        self.logger.info('Gamma: ' + str(self.gamma))
        self.logger.info('Alpha: ' + str(self.alpha))

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #   pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    REWARD_FOR_MOVING = 0
    if self.move:
        REWARD_FOR_MOVING = -0.1

    game_rewards = {
        e.COIN_COLLECTED: 0,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: 0,  # idea: the custom event is bad
        e.MOVED_LEFT: REWARD_FOR_MOVING,
        e.MOVED_RIGHT: REWARD_FOR_MOVING,
        e.MOVED_UP: REWARD_FOR_MOVING,
        e.MOVED_DOWN: REWARD_FOR_MOVING,
        e.WAITED: REWARD_FOR_MOVING,
        e.KILLED_SELF: 0,
        e.INVALID_ACTION :-20,
        e.BOMB_DROPPED :REWARD_FOR_MOVING,
        e.BOMB_EXPLODED :0,
        e.CRATE_DESTROYED:0,
        e.COIN_FOUND:0,
        e.GOT_KILLED : 0,
        e.OPPONENT_ELIMINATED : 0,
        e.SURVIVED_ROUND: 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum