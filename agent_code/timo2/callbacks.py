import os
import pickle
import random
from collections import deque
import numpy as np
from random import shuffle
import settings as s
#from .train import symmetry_x_y
#from .train import symmetry_up_down
#from .train import symmetry_left_right
from scipy.sparse import dok_matrix, coo_matrix
from scipy import sparse
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

    ##np_load_old = np.load
    ## modify the default parameters of np.load
    ##np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    ## call load_data with allow_pickle implicitly set to true
    #np_load_old = np.load
    #np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

    ## call load_data with allow_pickle implicitly set to true
    #matrix = sparse.load_npz("q_values.npz")
    #q_values = matrix.todok()
    ## restore np.load for future normal usage
    ## np.load.__defaults__=(None, False, True, 'ASCII')
    #np.load = np_load_old
    #print("File was loaded")
    ## restore np.load for future normal usage
    ##np.load = np_load_old

    # self.q_values = q_values
    self.rounds = 0
    self.random_prob = 0.4
    self.ruled_based_agent = False
    self.move = True
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.steps = 0
    self.total_steps = 0
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    #if self.train or not os.path.isfile("my-saved-model.pt"):
    #    self.logger.info("Setting up model from scratch.")
    #    weights = np.random.rand(len(ACTIONS))
    #    self.model = weights / weights.sum()
    #else:
    #    self.logger.info("Loading model from saved state.")
    #    with open("my-saved-model.pt", "rb") as file:
    #        self.model = pickle.load(file)
"""def number_of_state(u,d,l,r,m1x,m1y, m2x,m2y):
    sum = u + d*2 + l*2*2 + r*2*2*2 + m1x*2*2*2*2 + m1y*11*2*2*2*2 + m2x*11*11*2*2*2*2  + m2y*11*11*11*2*2*2*2
    return sum"""
def symmetry_up_down(features):
    m1x,m1y,free_spaces_complete, bomb1x, bomb1y,bomb_timer1 = features

    if m1y==5:
        m1y=-5
    if bomb1y==5:
        bomb1y=-5
    free_spaces_complete = np.flipud(free_spaces_complete.T).T
    return (m1x, -m1y,free_spaces_complete, bomb1x, -bomb1y,bomb_timer1)

def action_symmetry_up_down(action):
    if action == 0:
        action = 2
    if action == 2:
        action = 0
    return action

def symmetry_left_right(features):
    m1x,m1y,free_spaces_complete, bomb1x, bomb1y,bomb_timer1 = features
    if m1x==5:
        m1x=-5
    if bomb1x==5:
        bomb1x=-5
    free_spaces_complete = np.fliplr(free_spaces_complete.T).T

    return (-m1x, m1y,free_spaces_complete, -bomb1x, bomb1y,bomb_timer1)

def action_symmetry_left_right(action):
    if action ==1:
        action = 3
    if action ==3:
        action = 1
    return action

def symmetry_x_y(features):
    m1x,m1y,free_spaces_complete, bomb1x, bomb1y,bomb_timer1 = features
    free_spaces_complete = np.rot90(np.fliplr(free_spaces_complete.T)).T
    return (m1y, m1x,free_spaces_complete, bomb1y, bomb1x, bomb_timer1)
def action_symmetry_x_y(action):
    if action ==2:
        action = 1
    if action ==1:
        action = 2
    if action ==0:
        action = 3
    if action ==0:
        action = 3
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
        next_max = np.max(self.q_values[features_to_state_number(next_features)].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[features_to_state_number(features), action] = new_value
        # if old_value==0:
            # print("Old Value: " + str(old_value))
            # print("New Value: " + str(new_value))

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    #(d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed ]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    #print(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))].toarray())
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
        next_max = np.max(self.q_values[features_to_state_number(next_features)].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action] = new_value

    # Symmetry 1, interchange of up and down
    state_number_changed = features_to_state_number(symmetry_up_down(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_up_down(next_features))].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 2, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(features))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_left_right(action)
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(self.q_values[features_to_state_number(symmetry_left_right(next_features))].toarray())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[state_number_changed, action_changed] = new_value

    # Symmetry 3, interchange of lef and right:
    state_number_changed = features_to_state_number(symmetry_left_right(symmetry_up_down(features)))
    # (d, u, l, r, m1x, -m1y, changed_n, near, distance, state)
    action_changed = action_symmetry_up_down(action_symmetry_left_right(action))
    old_value = self.q_values[state_number_changed, action_changed]
    if type(old_value) == type(np.float64(0.3)):
        next_max = np.max(
            self.q_values[features_to_state_number(symmetry_left_right(symmetry_up_down(next_features)))].toarray())
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


def direction(a):
    x_direction = a[0]<=0
    y_direction = a[1]<=0
    if x_direction and y_direction:
        return 1
    if (not x_direction) and y_direction:
        return 2
    if (not x_direction) and (not y_direction):
        return 3
    if (x_direction) and (not y_direction):
        return 4

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
        coins = np.array(coins)
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        for (xc, yc) in coins:
            arena[xc, yc] = 3
        for (xa, ya) in others:
            arena[xa, ya] = 4
        for (xb, yb), t in bombs:
            arena[xb, yb] = t + 1 + 4
        r = 4
        (a, b) = arena.shape
        a = a + r * 2
        b = b + r * 2
        bigger_field = np.ones((a, b)) * 9
        bigger_field[r:-r, r:-r] = arena
        x_bigger = x + r
        y_bigger = y + r
        near_surrounding = np.ones((r * 2 + 1, r * 2 + 1))
        near_surrounding = bigger_field[x_bigger - r: x_bigger + r + 1, y_bigger - r: y_bigger + r + 1]

        u = 0
        d = 0
        l = 0
        r = 0
        m1x = 5
        m1y = 5
        bomb1x = 5
        bomb1y = 5
        bomb2x = 5
        bomb2y = 5
        bomb_timer1=0
        agent1x = 5
        agent2x = 5
        agent1y = 5
        agent2y = 5
        bomb_timer2 = 0
        #crate_array = np.zeros(16)

        #n = 0
        #near= 0
        #distance = 0

        ## QUE HICISTE AQUI?
        """if arena[(x,y+1)] == 0:
            d = 1
        if arena[(x,y-1)] == 0:
            u = 1
        if arena[(x + 1,y)] == 0:
            r = 1
        if arena[(x - 1,y)] == 0:
            l = 1"""
        if arena[(x, y + 1)] == 1:
            d = 1
        if arena[(x, y - 1)] == 1:
            u = 1
        if arena[(x + 1, y)] == 1:
            r = 1
        if arena[(x - 1, y)] == 1:
            l = 1

        """for i in range(5):
            if bigger_field[(x - 2 + i, y - 2)] == 1:
                crate_array[i] = 1

        for i in range(4):
            if bigger_field[(x + 2, y - 1 + i)] == 1:
                crate_array[i + 5] = 1

        for i in range(4):
            if bigger_field[(x + 1 - i, y + 2)] == 1:
                crate_array[i + 9] = 1

        for i in range(3):
            if bigger_field[(x - 2, y + 1 - i)] == 1:
                crate_array[i + 13] = 1"""

        """if bigger_field[(x - 3, y)] == 1:
            crate_array[16] = 1

        if bigger_field[(x, y - 3)] == 1:
            crate_array[17] = 1


        if bigger_field[(x + 3, y)] == 1:
            crate_array[18] = 1


        if bigger_field[(x, y + 3)] == 1:
            crate_array[19] = 1"""

        free_spaces_complete = np.where(near_surrounding==0, 1,0)

        #distance = bomb_map
        #distance[:,] = distance[:,]  - x
        #distance[,: ] = distance[,: ] - y
        max_distance = 4     # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES
        possible_outcomes = max_distance *2 + 1
        """if len(coins)>=2:
            coins_position = coins - (x,y)
            coins_distance = np.sum(np.abs(coins_position), axis=1)
            first = np.argmin(coins_distance)
            coins_distance[first] = 0
            second = np.argmin(coins_distance)
            coins_distance[second] = 0
            index = np.array([first,second])
            coins_position_near = coins_position[index]
            coins_position_near = coins_position_near[
                np.all([np.abs(coins_position_near[:, 0] - x) <= max_distance, np.abs(coins_position_near[:, 1] - y) <= max_distance], axis=0)]
            if len(coins_position_near)>=1:
                near_coins = 1
                m1x = coins_position[first][0] +max_distance
                m1y = coins_position[first][1] +max_distance
                #near = 1
                #distance = np.abs(m1x) + np.abs(m1y)
                #m2y = max_distance+1
                #m2x = max_distance+1
                #n = 0 #direction(coins_position[second])

            #if len(coins_position_near)== 0:
                #near = 0
                #m1x = max_distance+1
                #m1y = max_distance + 1
                #m2x = max_distance + 1
                #m2y = max_distance + 1
                #n = direction(coins_position[first])
                #distance = np.sum(np.abs(coins_position[first]))

        #print(x,y)
        if len(coins) == 1:
            coins_position = coins - (x, y)
            m1x = coins_position[0][0] +max_distance
            m1y = coins_position[0][1] +max_distance"""
        """if np.abs(m1x)<= max_distance and np.abs(m1y)<=max_distance:
                near = 1
                # m2y = max_distance+1
                # m2x = max_distance+1
                distance = np.abs(m1x) + np.abs(m1y)
                n = 0
            else:
                near = 0
                n = direction(coins_position[0])
                distance = np.abs(m1x) + np.abs(m1y)"""


        a, b = np.where(near_surrounding == 3)
        position_of_coins = np.array([])
        position_of_coins = np.append([position_of_coins], [a])
        position_of_coins = np.append([position_of_coins], [b], axis=0)
        position_of_coins = position_of_coins - 4
        distance_coins = np.sum(np.abs(position_of_coins), axis=0)
        if len(distance_coins) >= 1:
            first = np.argmin(distance_coins)
            m1x = position_of_coins[0][first]
            m1y = position_of_coins[1][first]



        a,b = np.where(np.all([near_surrounding>=5, near_surrounding<9],axis = 0))
        position_of_bombs = np.array([])
        position_of_bombs = np.append([position_of_bombs], [a])
        position_of_bombs = np.append([position_of_bombs], [b], axis=0)
        position_of_bombs = position_of_bombs - 4
        distance_bombs = np.sum(np.abs(position_of_bombs), axis=0)
        if len(distance_bombs) >= 1:
            first = np.argmin(distance_bombs)
            bomb1x = position_of_bombs[0][first]
            bomb1y = position_of_bombs[1][first]
            bomb_timer1 = near_surrounding[a[first],b[first]] -4
        """if len(distance_bombs ) >= 2:
            first = np.argmin(distance_bombs )
            distance_bombs [first] = 100
            second = np.argmin(distance_bombs)
            bomb1x = position_of_bombs[0][first]
            bomb1y = position_of_bombs[1][first]
            bomb2x = position_of_bombs[0][second]
            bomb2y = position_of_bombs[1][second]
            bomb_timer1 = near_surrounding[a[first], b[first]] -4
            bomb_timer2 = near_surrounding[a[second], b[second]] -4"""

        """a, b = np.where(near_surrounding == 4)
        position_of_agent = np.array([])
        position_of_agent = np.append([position_of_agent], [a])
        position_of_agent = np.append([position_of_agent], [b], axis=0)
        position_of_agent = position_of_agent - 4
        distance_of_agent = np.sum(np.abs(position_of_agent), axis=0)
        if len(distance_of_agent) == 1:
            first = np.argmin(distance_of_agent)
            agent1x = position_of_agent[0][first]
            agent1y = position_of_agent[1][first]
        if len(distance_of_agent) >= 2:
            first = np.argmin(distance_of_agent)
            distance_of_agent[first] = 100
            second = np.argmin(distance_of_agent)
            agent1x = position_of_agent[0][first]
            agent1y = position_of_agent[1][first]
            agent2x = position_of_agent[0][second]
            agent2y = position_of_agent[1][second]"""
        """m1x= m1x 
        m1y = m1y + max_distance
        bomb1x = bomb1x + max_distance
        bomb1y = bomb1y + max_distance
        bomb2x = bomb2x+ max_distance
        bomb2y = bomb2y + max_distance
        agent1x = agent1x + max_distance
        agent1y= agent1y + max_distance
        agent2x= agent2x + max_distance
        agent2y = agent2y + max_distance"""





        #state = True
        return (u,d,l,r,m1x,m1y,free_spaces_complete, bomb1x, bomb1y,bomb_timer1)
    if game_state is None:
        return None


def features_to_state_number(features):
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

    if features is not None:
        u,d,l,r,m1x,m1y,free_spaces_complete, bomb1x, bomb1y,bomb_timer1= features
        max_distance = 4     # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES
        free_spaces = np.zeros(41)
        free_spaces[0] = free_spaces_complete.T[0, 4]
        free_spaces[1:4] = free_spaces_complete.T[1, 3:6]
        free_spaces[4:9] = free_spaces_complete.T[2, 2:7]
        free_spaces[9:16] = free_spaces_complete.T[3, 1:8]
        free_spaces[16:25] = free_spaces_complete.T[4, :]
        free_spaces[25:32] = free_spaces_complete.T[5, 1:8]
        free_spaces[32:37] = free_spaces_complete.T[6, 2:7]
        free_spaces[37:40] = free_spaces_complete.T[7, 3:6]
        free_spaces[40] = free_spaces_complete.T[8, 4]
        #print(free_spaces_complete.T)
        #print(free_spaces)
        m1x = m1x + max_distance
        m1y = m1y + max_distance
        bomb1x = bomb1x + max_distance
        bomb1y = bomb1y + max_distance
        possible_outcomes = max_distance *2 + 1
        flattened_features = np.array([u,d,l,r,m1x,m1y, bomb1x, bomb1y, bomb_timer1],dtype = np.int64)
        number_of_values_per_feature = np.array([2,2,2,2,10, 10, 10, 10, 5], dtype=np.int64)
        for i in range(len(free_spaces)):
            flattened_features = np.append(flattened_features, free_spaces[i])
            number_of_values_per_feature = np.append(number_of_values_per_feature,2)
        number_of_state = 0
        multiplier = 1
        for i in range(len(flattened_features)):
            number_of_state = number_of_state + flattened_features[i] * multiplier
            multiplier = multiplier * number_of_values_per_feature[i]
        return number_of_state



    # This is the dict before the game begins and after it ends
    if features == None:
        print("game state is none")
        return None

def state_finder(game_state):
    return features_to_state_number(state_to_features(game_state))

### CODE FROM RULED_BASED_AGENT FOR TRAINING

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def act2(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
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

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, s.COLS-1) for y in range(1, s.COLS-1) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, s.COLS-1) for y in range(1, s.COLS-1) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a




def act(self, game_state):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #self.q_values = np.load('q_values.npy')
    #_, score, bombs_left, (x, y) = game_state['self']
    #print(x,y)
    self.logger.info('Picking action according to rule set')
    #print(os.path.dirname("q_values.npz"))
    #print(state_finder(game_state))
    #features = state_to_features(game_state)
    #print(state_to_features(game_state))
    #print(symmetry_up_down(features))
    #print(symmetry_left_right(features))
    #print(symmetry_x_y(features))
    #print(symmetry_left_right(symmetry_up_down(features)))
    #features = symmetry_x_y(features)
    #print(symmetry_up_down(features))
    #print(symmetry_left_right(features))
    #print(symmetry_left_right(symmetry_up_down(features)))
    #print(self.steps)
    # Check if we are in a different round
    # todo Exploration vs exploitation
    self.steps = self.steps + 1
    self.total_steps = self.total_steps + 1
    #print(state_to_features(game_state))
    #print(state_finder(game_state))
    #random_prob =0.2#### ESTO SE DEBE CAMBIAR
    #ruled_based_agent = False
    if self.train and random.random() < self.random_prob:
        #print("random")
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return  np.random.choice(ACTIONS, p=[1/6,1/6,1/6,1/6,1/6,1/6])      #[0.25, 0.25, 0.25, 0.25, 0, 0]
    if not self.train:
        action = np.argmax(self.q_values[state_finder(game_state)].toarray())
        # print(action)
        if action == None:
            return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
        if action == 0:
            return "UP"
        if action == 1:
            return "RIGHT"
        if action == 2:
            return "DOWN"
        if action == 3:
            return "LEFT"
        if action == 4:
            return "WAIT"
        if action == 5:
            return "BOMB"
    else:
        if not self.ruled_based_agent:
            if self.total_steps<=1000:
                if state_finder(game_state) !=None:
                    #if state_finder(game_state) >196080:
                        #print("BRO, ESTE FEATURE ESTA RARO: " + str(state_finder(game_state)))
                        #print(game_state)
                #print("Action")
                    #print(state_finder(game_state))
                    action = np.argmax(self.q_values[state_finder(game_state)].toarray())
                    #print(action)
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
            else:
                return act2(self, game_state)
        if self.ruled_based_agent:
            return act2(self, game_state)
    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)



