import os
import pickle
import random
from collections import deque
import numpy as np


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
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    #if self.train or not os.path.isfile("my-saved-model.pt"):
    #    self.logger.info("Setting up model from scratch.")
    #    weights = np.random.rand(len(ACTIONS))
    #    self.model = weights / weights.sum()
    #else:
    #    self.logger.info("Loading model from saved state.")
    #    with open("my-saved-model.pt", "rb") as file:
    #        self.model = pickle.load(file)
def number_of_state(u,d,l,r,m1x,m1y, m2x,m2y):
    sum = u + d*2 + l*2*2 + r*2*2*2 + m1x*2*2*2*2 + m1y*11*2*2*2*2 + m2x*11*11*2*2*2*2  + m2y*11*11*11*2*2*2*2
    return sum

def state_to_features(game_state):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    #self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    # Gather information about the game state
    if isinstance(game_state, dict):
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        u = 0
        d = 0
        l = 0
        r = 0

        if y+1 == 0:
            u = 1
        if y-1 == 0:
            d = 1
        if x + 1 == 0:
            r = 1
        if x - 1 == 0:
            l = 1
        #distance = bomb_map
        #distance[:,] = distance[:,]  - x
        #distance[,: ] = distance[,: ] - y
        coins_position = deque([],2)
        n = 0
        #print(x,y)
        for coin in coins:

            (a,b) = coin
            if np.abs(a-x) <=5 and np.abs(b-y)<=5:
                coins_position.append((a-x,b-y))
                n = n + 1
        if n==1:
            coins_position.append((-100,-100))
        if n == 0:
            coins_position.append((-100, -100))
            coins_position.append((-100, -100))
        m1x,m1y = coins_position[0]
        m2x,m2y = coins_position[1]
        #print(m1x)
        #print(m2x)
        #print(m1y)
        #print(m2y)
        if n ==1:
            return 2*2*2*2*11*11*11*11+ u + d*2 + l*2*2 + r*2*2*2 + (m1x+5)*2*2*2*2 + (m1y+5)*11*2*2*2*2
        if n ==0:
            return 2*2*2*2*11*11*11*11 + 2*2*2*2*11*11 +u + d*2 + l*2*2 + r*2*2*2
        if n == 2:
            return number_of_state(u,d,l,r,m1x+5,m1y+5, m2x+5,m2y+5)



    # This is the dict before the game begins and after it ends
    if game_state is None:
        print("game state is none")
        return None

    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)

    # concatenate them as a feature tensor (they must have the same shape), ...

    #stacked_channels = np.stack(channels)

    # and return them as a vector
    #return stacked_channels.reshape(-1)


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def act(self, game_state):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #self.q_values = np.load('q_values.npy')
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    # todo Exploration vs exploitation
    print(state_to_features(game_state))
    random_prob = 0 ##### ESTO SE DEBE CAMBIAR
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    else:
        action = np.argmax(self.q_values[state_to_features(game_state)])
        if action == None:
            return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
        if action ==0:
            return "UP"
        if action ==1:
            return "RIGHT"
        if action ==2:
            return "DOWN"
        if action ==3:
            return "LEFT"
        if action == 4:
            return "WAIT"
        if action == 5:
            return "BOMB"
        else:
            return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)



