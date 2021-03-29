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
TRANSITION_HISTORY_SIZE = 4 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXTRA_REWARD = 0.4
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
    self.distance = deque(maxlen=3)
    self.repition = deque(maxlen=3)
    self.distance_rewarder = deque(maxlen =3)
    #self.samefeature = deque(maxlen=REPETION_SEARCH)
    self.states = np.array([])
    self.coins_collected = REPETION_SEARCH
    max_distance = 3  # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES, TAMBIEN EN CALLBACKS
    possible_outcomes = max_distance * 2 + 1
    number_of_space_taker = 3
    row = np.array([])
    col = np.array([])
    data = np.array([])
    self.alpha = 0.2
    self.gamma = 0.2
    r = 4
    q_values = np.zeros((256, 6))
    q_values[:,4] = -1000
    q_values[:,5] = -1000
    # q_values = np.ones((201411,6))
    if self.train and os.path.isfile("q_values.npy"):
        q_values = np.load("q_values.npy")
        print("File was loaded")
    self.q_values = q_values

def symmetry_up_down(features):
    u,d,l,r,m1x,m1y,x_better, distance = features
    #free_spaces_complete = np.flipud(free_spaces_complete.T).T
    return (d, u, l, r,m1x, -m1y,x_better, distance)

def action_symmetry_up_down(action):
    if action == 0:
        return 2
    if action == 2:
        return 0
    else:
        return action

def symmetry_left_right(features):
    u,d,l,r,m1x,m1y, x_better, distance = features
    #free_spaces_complete = np.fliplr(free_spaces_complete.T).T

    return (u, d, r, l,-m1x, m1y,x_better, distance)

def action_symmetry_left_right(action):
    if action ==1:
        return 3
    if action ==3:
        return 1
    else:
        return action

def symmetry_x_y(features):
    u,d,l,r,m1x,m1y, x_better, distance= features
    if x_better ==0:
        x_better =1
    if x_better ==1:
        x_better = 0
    #free_spaces_complete = np.rot90(np.fliplr(free_spaces_complete.T)).T
    return (l, r, u, d,m1y, m1x,x_better, distance)
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

def use_symmetry(features, action, next_features, reward, self):
    if features is None or next_features is None:
        return
    old_value = self.q_values[features_to_state_number(features), action]
    #print("Old Value: " + str(old_value) +str(type(old_value)))
    #print(type(np.float64(0.3)))
    #print(type(old_value) == type(np.float64(0.3)))
    #print(action)
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(next_features)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        self.q_values[features_to_state_number(features), action] = new_value
        #if old_value==0:
            #print("Old Value: " + str(old_value))
            #print("New Value: " + str(new_value))

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    #(d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed ]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    #print(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 4, interchange of x and y:
    features = symmetry_x_y(features) #Ultima vez que usamos todos estos valores, entonces podemos cambiarlo por facilidad
    next_features = symmetry_x_y(next_features)
    state_number_changed = features_to_state_number(features)
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action = action_symmetry_x_y(action)
    old_value = self.q_values[state_number_changed, action]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(next_features)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action] = new_value

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

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
        u,d,l,r,m1x,m1y, x_better,distance_coin =state_to_features(old_game_state)
        self.distance.append(distance_coin)
        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
        self.distance_rewarder.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
        self.repition.append(state_finder(old_game_state))
        repition = True


        #print(symmetry_x_y(state_to_features(old_game_state)))
        #print(symmetry_up_down(state_to_features(old_game_state)))
        #print(symmetry_left_right(state_to_features(old_game_state)))

        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
        # print(events)
        features = state_to_features(old_game_state)
        next_features = state_to_features(new_game_state)
        next_state = features_to_state_number(next_features)
        state = features_to_state_number(features)

        numpy_events = np.array(events)




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
            last_transition = self.transitions.pop()
            self.distance_rewarder.pop()
            #self.distance.pop()
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
                        reward = reward - 0.01
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                        self.repition.pop()
                        self.distance.pop()
                        self.steps = self.steps - 1
            if len(self.repition) == 5:
                if np.all(self.repition[0] == self.repition[5]):
                    #print("Repetion was made #2")
                    while self.repition:
                        features, action, next_features, reward = self.transitions.pop()
                        reward = reward - 0.01
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                        self.repition.pop()
                        self.distance.pop()
                        self.steps = self.steps - 1

                self.repition.clear()
                # repition = False
                #print("Repition was cleared")"""

        # state_to_features is defined in callbacks.py
        if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            coin_colected = True
            while self.transitions:
                features, action, next_features, reward = self.transitions.pop()
                reward = reward + EXTRA_REWARD
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
            self.distance.clear()
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
            self.distance.clear()
            self.transitions.clear()
            self.repition.clear()
            #print("0 Coins were collected and Q-Table got updated")
            self.coins_collected = 0
            self.steps = 0
    distance_rewards = True
    if self.distance and self.transitions:
        if len(self.distance) == 3 and len(self.transitions) >= 3:
            if not np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
                if self.distance[0] < self.distance[2] and self.distance_rewarder:
                    self.logger.info("Got farther away")
                    distance_rewards = False
                    while self.distance_rewarder:
                        features, action, next_features, reward = self.distance_rewarder.pop()
                        self.steps = self.steps - 1
                        reward = reward - 5
                        action = action_number(action)
                        use_symmetry(features, action, next_features, reward, self)
                    self.distance.clear()
                    self.repition.clear()
                    self.distance_rewarder.clear()
                if distance_rewards:
                    if self.distance[0] > self.distance[2] and self.distance_rewarder:
                        self.logger.info("Got closer")
                        while self.distance_rewarder:
                            features, action, next_features, reward = self.distance_rewarder.pop()
                            self.steps = self.steps - 1
                            reward = reward + 5
                            action = action_number(action)
                            use_symmetry(features, action, next_features, reward, self)
                        self.distance.clear()
                        self.repition.clear()
                        self.distance_rewarder.clear()
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
    self.rounds = self.rounds + 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
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
    if self.rounds%100 ==0:
        print("file was saved")
        self.logger.debug(f'File was saved')
        np.save("q_values.npy", self.q_values)
        #self.ruled_based_agent = not self.ruled_based_agent
        #self.move = not self.move
        self.random_prob = self.random_prob * 0.9
    #if self.rounds % 3500 == 0:
        #self.random_prob = self.random_prob * 0.9
    if self.rounds % 10000==0:
        self.alpha = self.alpha * 0.999
        self.gamma = self.gamma * 0.999
    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #   pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    REWARD_FOR_MOVING = -0.1
    if self.move:
        REWARD_FOR_MOVING = -0.1

    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        e.MOVED_LEFT: REWARD_FOR_MOVING,
        e.MOVED_RIGHT: REWARD_FOR_MOVING,
        e.MOVED_UP: REWARD_FOR_MOVING,
        e.MOVED_DOWN: REWARD_FOR_MOVING,
        e.WAITED: -1,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION :-20,
        e.BOMB_DROPPED :REWARD_FOR_MOVING,
        e.BOMB_EXPLODED :0.2,
        e.CRATE_DESTROYED:0.3,
        e.COIN_FOUND:0,
        e.GOT_KILLED : -5,
        e.OPPONENT_ELIMINATED : 0.2,
        e.SURVIVED_ROUND: 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum