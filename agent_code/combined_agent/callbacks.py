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

## CODE FROM THE SURVIVER
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
    self.rounds = 0
    self.random_prob = 0
    self.ruled_based_agent = False
    self.move = True
    self.alpha = 0.2
    self.gamma = 0.2
    self.q_values_crates = np.load("q_values_crates.npy")
    self.q_values_coins = np.load("q_values_coins.npy")
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


def state_to_features_crates(game_state):
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
        bomb_timer1 = 0
        b1x = 0
        b2x = 0
        next_to_crate = 0 #ESTO FUE AGREGADO
        r = 17 #ESTO FUE AGREGADO
        if arena[(x, y + 1)] == 1 or arena[(x, y - 1)] == 1 or arena[(x + 1, y)] == 1 or arena[(x - 1, y)] == 1:
           next_to_crate = 1
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)
                    #arena[i, j] = bomb_map[i,j]
        distance_bombs = 10
        for (xb, yb), t in bombs:
            arena[xb, yb] = 12+t
            #bomb_timer1 = t+1
            b1x = xb + r
            b2x = yb + r
            distance_actual = np.abs(xb - x) + np.abs(yb - y)
            if np.abs(xb - x) + np.abs(yb - y) < 4 and distance_actual<distance_bombs:
                bomb_timer1 = t + 1
                distance_bombs = distance_actual
        for (xc,yc) in coins:
            arena[xc,yc]=0
        arena[x,y] =-5


        (a, b) = arena.shape
        a = a + r * 2
        b = b + r * 2
        bigger_field = np.ones((a, b)) * 9
        bigger_field[r:-r, r:-r] = arena
        x_bigger = x + r
        y_bigger = y + r
        direction_of_nearest_crate =4 #ESTO FUE AGREGADO
        up = True
        right = True
        down = True
        left = True
        for i in range(15):  #ESTO FUE AGREGADO
            if bigger_field.T[y_bigger, x_bigger+i]==1 and right:
                direction_of_nearest_crate = 1
                break
            if bigger_field.T[y_bigger, x_bigger + i] == -1:
                right = False
            if bigger_field.T[y_bigger, x_bigger-i]==1 and left:
                direction_of_nearest_crate = 3
                break
            if bigger_field.T[y_bigger, x_bigger - i] == -1:
                left= False

            if bigger_field.T[y_bigger + i, x_bigger]==1 and down:
                direction_of_nearest_crate = 2
                break
            if bigger_field.T[y_bigger + i, x_bigger] == -1:
                down = False
            if bigger_field.T[y_bigger-i, x_bigger]==1 and up:
                direction_of_nearest_crate = 0
                break
            if bigger_field.T[y_bigger - i, x_bigger] == -1:
                up = False
        r = 4  # ESTO FUE AGREGADO
        near_surrounding = np.ones((r * 2 + 1, r * 2 + 1))
        near_surrounding = bigger_field[x_bigger - r: x_bigger + r + 1, y_bigger - r: y_bigger + r + 1]
        a, b = np.where(near_surrounding >= 12)
        position_of_bombs = np.array([])
        position_of_bombs = np.append([position_of_bombs], [a])
        position_of_bombs = np.append([position_of_bombs], [b], axis=0)
        position_of_bombs = position_of_bombs - 4
        distance_bombs = np.sum(np.abs(position_of_bombs), axis=0)
        if len(distance_bombs) >= 1:
            r = 6
            first = np.argmin(distance_bombs)
            b1x = np.int64(position_of_bombs[0][first] +x_bigger)
            #print("b1x")
            #print(b1x)
            b2x = np.int64(position_of_bombs[1][first]  + y_bigger)
            #print(b2x)
            #print(bigger_field[b1x - r: b1x + r + 1, b2x - r: b2x + r + 1].T)
            bomb_timer1 = near_surrounding[a[first], b[first]] - 11

        danger = 0
        if len(bombs)>=1 and len(a)>0:
            #bomb_timer1 = bigger_field[a,b] -11
            if b1x != x_bigger or b2x !=y_bigger:
                for i in range(3):
                    x_bomb = b1x+1+i
                    if bigger_field.T[b2x, x_bomb]==-5:
                        danger = 1
                        break
                    if bigger_field.T[b2x, x_bomb]==-1:
                        danger = 1
                        break
                    bigger_field.T[b2x, x_bomb] = bomb_timer1
                for i in range(3):
                    x_bomb = b1x -1 -i
                    if bigger_field.T[b2x, x_bomb]==-5:
                        danger = 1
                        break
                    if bigger_field.T[b2x, x_bomb]==-1:
                        danger = 1
                        break
                    bigger_field.T[b2x, x_bomb] = bomb_timer1

                for i in range(3):
                    y_bomb = b2x -1 - i
                    if bigger_field.T[y_bomb, b1x]==-5:
                        danger = 1
                        break
                    if bigger_field.T[y_bomb, b1x]==-1:
                        danger = 1
                        break
                    bigger_field.T[y_bomb, b1x] = bomb_timer1
                for i in range(3):
                    y_bomb = b2x + i +1
                    if bigger_field.T[y_bomb, b1x] == -5:
                        danger = 1
                        break
                    if bigger_field.T[y_bomb, b1x] == -1:
                        danger = 1
                        break
                    bigger_field.T[y_bomb, b1x] = bomb_timer1
        arena[x, y] = 0

        r = 4 #ESTO FUE AGREGADO
        near_surrounding = np.ones((r * 2 + 1, r * 2 + 1))
        near_surrounding = bigger_field[x_bigger - r: x_bigger + r + 1, y_bigger - r: y_bigger + r + 1]

        u = 0
        d = 0
        l = 0
        r = 0
        m1x = 0
        m1y = 0
        bomb1x = 5
        bomb1y = 5

        free_spaces_complete = np.where(near_surrounding==0, 1,0)


        #state = True
        return (next_to_crate, free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate)
    if game_state is None:
        return None


def features_to_state_number_crates(features):
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
        next_to_crate, free_spaces_complete,bomb_timer1,danger,direction_of_nearest_crate= features #ESTO FUE AGREGADO
        max_distance = 4     # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES
        """free_spaces = np.zeros(41)
        free_spaces[0] = free_spaces_complete.T[0, 4]
        free_spaces[1:4] = free_spaces_complete.T[1, 3:6]
        free_spaces[4:9] = free_spaces_complete.T[2, 2:7]
        free_spaces[9:16] = free_spaces_complete.T[3, 1:8]
        free_spaces[16:25] = free_spaces_complete.T[4, :]
        free_spaces[25:32] = free_spaces_complete.T[5, 1:8]
        free_spaces[32:37] = free_spaces_complete.T[6, 2:7]
        free_spaces[37:40] = free_spaces_complete.T[7, 3:6]"""
        #free_spaces[40] = free_spaces_complete.T[8, 4]
        #print(free_spaces_complete.T)
        #print(free_spaces)
        #print(free_spaces_complete.T)
        #direction_coin = direction((m1x,m1y))
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

        for i in range(xm): #ESTO FUE AGREGAD0
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
        possible_outcomes = max_distance *2 + 1
        flattened_features = np.array([next_to_crate,ym, yp,xm,xp,ymn, ypn,xmn,xpn,bomb_timer1,danger,direction_of_nearest_crate],dtype = np.int64)
        number_of_values_per_feature = np.array([2,5,5,5,5,2,2,2,2,5,2,5], dtype=np.int64)#ESTO FUE AGREGADO
        number_of_state = 0
        multiplier = 1
        for i in range(len(flattened_features)):
            number_of_state = number_of_state + flattened_features[i] * multiplier
            multiplier = multiplier * number_of_values_per_feature[i]
        return number_of_state



    # This is the dict before the game begins and after it ends
    if features == None:
        #print("game state is none")
        return None

def state_finder_crates(game_state):
    return features_to_state_number_crates(state_to_features_crates(game_state))

### CODE FROM THE COIN COLLECTER
def direction(a):
    right = a[0]>0
    up = a[1]<0
    left = a[0]<0
    down = a[1]>0
    vertical = a[0] == 0
    horizontal = a[1] == 0
    if right and up:
        return 0
    if left and up:
        return 1
    if left and down:
        return 2
    if right and down:
        return 3
    if horizontal and right:
        return 4
    if horizontal and left:
        return 5
    if vertical and up:
        return 6
    if vertical and down:
        return 7
    if vertical and horizontal:
        return 8
def state_to_features_coins(game_state):
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
                    arena[i, j] = bomb_map[i,j]
        for (xb, yb), t in bombs:
            arena[xb, yb] = 12+t
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
        m1x = 0
        m1y = 0
        #n = 0
        #near= 0
        #distance = 0

        ## QUE HICISTE AQUI?
        if arena[(x,y+1)] == -1:
            d = 1
        if arena[(x,y-1)] == -1:
            u = 1
        if arena[(x + 1,y)] == -1:
            r = 1
        if arena[(x - 1,y)] == -1:
            l = 1

        max_distance = 4     # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES
        distance_coin = 0
        x_better = 0
        if len(coins)>=1:
            coins_position = coins - (x,y)
            coins_distance = np.sum(np.abs(coins_position), axis=1)
            first = np.argmin(coins_distance)
            m1x = coins_position[first][0]
            m1y = coins_position[first][1]
            distance_coin = np.abs(m1x) + np.abs(m1y)
            direction_coin = direction((m1x, m1y))
            if direction_coin != 6 and direction_coin != 7:
                if np.abs(m1x) > np.abs(m1y):
                    x_better = 1
                if np.abs(m1x) == np.abs(m1y):
                    x_better = 2
        return (u,d,l,r,m1x,m1y,x_better,distance_coin)
    if game_state is None:
        return None


def features_to_state_number_coins(features):
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
        u,d,l,r,m1x,m1y, x_better,distance_coin= features
        max_distance = 4     # PUES CAMBIAR ESTO PARA RESULTADOS DIFERENTES
        direction_coin = direction((m1x,m1y))
        possible_outcomes = max_distance *2 + 1
        flattened_features = np.array([u,d,l,r, direction_coin , x_better],dtype = np.int64)
        number_of_values_per_feature = np.array([2,2,2,2,9,3], dtype=np.int64)
        number_of_state = 0
        multiplier = 1
        for i in range(len(flattened_features)):
            number_of_state = number_of_state + flattened_features[i] * multiplier
            multiplier = multiplier * number_of_values_per_feature[i]
        return number_of_state



    # This is the dict before the game begins and after it ends
    if features == None:
        #print("game state is none")
        return None

def state_finder_coins(game_state):
    return features_to_state_number_coins(state_to_features_coins(game_state))

def q_table_chooser(game_state):
    if isinstance(game_state, dict):
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        near_agents = False
        for (xa, ya) in others:
            if np.abs(xa-x)+np.abs(ya-x)<4:
                return 1
        coins = game_state['coins']
        coins = np.array(coins)
        coins_in_map = False
        numpy_bombs = np.array(bombs)
        bomb_in_map = len(numpy_bombs)!=0
        #print(bomb_in_map)
        if len(coins)>=1 and not bomb_in_map:
            coins_in_map  = True
            coins_position = coins - (x,y)
            coins_distance = np.sum(np.abs(coins_position), axis=1)
            first = np.argmin(coins_distance)
            m1x = coins_position[first][0]
            m1y = coins_position[first][1]
            direction_coin = direction((m1x, m1y))
            m1x = m1x+x
            m1y = m1y + y
            #print(arena.T)
            #print(x,m1x)
            #print(y,m1y)
            if direction_coin ==0:
                smaller_arena = arena.T[m1y:y+1, x:m1x+1]
                #print(smaller_arena)
            if direction_coin ==1:
                smaller_arena = arena.T[m1y:y+1, m1x:x+1]
                #print(smaller_arena)
            if direction_coin ==2:
                smaller_arena = arena.T[y:m1y+1, m1x:x+1]
                #print(smaller_arena)
            if direction_coin ==3:
                smaller_arena = arena.T[y:m1y+1, x:m1x+1]
                #print(smaller_arena)
            if direction_coin ==4:
                smaller_arena = arena.T[y,x:m1x+1]
                #print(smaller_arena)
            if direction_coin ==5:
                smaller_arena = arena.T[y, m1x:x+1]
                #print(smaller_arena)
            if direction_coin ==6:
                smaller_arena = arena.T[m1y:y+1, x]
                #print(smaller_arena)
            if direction_coin ==7:
                smaller_arena = arena.T[y:m1y+1, x]
                #print(smaller_arena)
            if np.any(smaller_arena==1):
                return 1
            if not np.any(smaller_arena==1):
                return 0
        return 1


def act(self, game_state):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if we are in a different round
    # todo Exploration vs exploitation
    self.steps = self.steps + 1
    self.total_steps = self.total_steps + 1
    q_table = q_table_chooser(game_state)
    if self.train and random.random() < self.random_prob:
        #print("random")
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return  np.random.choice(ACTIONS, p=[1/6,1/6,1/6,1/6,1/6,1/6])      #[0.25, 0.25, 0.25, 0.25, 0, 0]
    if not self.train:
        if q_table==0:
            self.logger.info("The coin collector was summoned")
            action = np.argmax(self.q_values_coins[state_finder_coins(game_state)])
        if q_table==1:
            self.logger.info("The crate destroyer was summoned")
            action = np.argmax(self.q_values_crates[state_finder_crates(game_state)])

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
        return np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)



