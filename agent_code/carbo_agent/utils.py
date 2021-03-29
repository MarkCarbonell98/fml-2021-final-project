import os
import random
import numpy as np
import unittest

def are_bombs_in_map(game_state: dict) -> bool:
    return len(game_state['bombs']) > 0

def is_agent_safe(game_state: dict) -> bool:
    self_x, self_y = game_state['self'][3]
    bomb_range = 3
    if are_bombs_in_map(game_state):
        # check if the player is within the explosion range of the bomb
        for bomb in game_state['bombs']:
            bomb_x, bomb_y, bomb_t = bomb
            intersect_x = self_x >= (bomb_x  + bomb_range) or self_x <= (bomb_x - bomb_range)
            intersect_y = self_y >= (bomb_y + bomb_range) or self_y <= (bomb_y - bomb_range)
            if intersect_x and intersect_y:
                return False
    return True

def decide(game_state: dict) -> str:
    if are_bombs_in_map(game_state): # check if there is a bomb or explosion range nearby the agent
        if is_agent_safe(game_state): # check for bombs and explosions
            return 'WAIT'
        else:
            safehouse_coords = find_safe_place(game_state)
            return move_agent_to(safehouse_coords) # up, down, right, left...
    else:
        if are_coins_in_map_reachable(game_state):
            nearest_coin_coords = find_nearest_coin(game_state)
            return move_agent_to(nearest_coin_coords) # up, down, right, left...
        else:
            if are_opponents_reachable(game_state):
                opponent_coords = find_nearest_opponent(game_state)
                if (is_opponent_adjacent(game_state) or is_opponent_locked(game_state)) and is_safe_to_drop_bomb(game_state):
                    return 'BOMB'
                else:
                    return move_agent_to(opponent_coords)
            else:
                nearest_crate_coords = find_nearest_crate(game_state)
                if is_crate_adjacent(game_state) and is_safe_to_drop_bomb(game_state):    return  'BOMB'
                else:
                    return move_agent_to(nearest_crate_coords)
    return 'WAIT'

def get_adjacent_values_to_coordinate(game_state: dict, coords: tuple) -> dict:
    start_x, start_y = coords
    field = game_state['field']
    map_x = field.shape[0]
    map_y = field.shape[1]
    print(field.shape, coords)
    adjacent_values = {
        'up': None,
        'down': None,
        'left': None,
        'right': None
    }
    if start_x > 0:
        adjacent_values['left'] = ((start_x - 1, start_y), field[start_x - 1 , start_y])
    if start_x < map_x:
        adjacent_values['right'] = ((start_x + 1, start_y), field[start_x + 1 , start_y])
    if start_y > 0:
        adjacent_values['up'] = ((start_x, start_y + 1), field[start_x, start_y + 1])
    if start_y < map_y:
        adjacent_values['down'] = ((start_x, start_y - 1), field[start_x, start_y - 1])
    return adjacent_values


def get_adjacent_values_to_player(game_state: dict) -> dict:
    self_x, self_y = game_state['self'][3] # start from 0
    return get_adjacent_values_to_coordinate(game_state, (self_x - 1, self_y - 1))


def are_crates_in_map(game_state: dict) -> bool:
    return (2 in game_state['field'])

def are_coins_in_map(game_state: dict) -> bool:
    return len(game_state['coins']) > 0

def are_opponents_alive(game_state: dict) -> bool:
    return len(game_state['others']) > 0

def is_coordinate_reachable(game_state, coords) -> bool:
    self_x, self_y = game_state['self'][3]
    # make a graph of the map
    # to A* or BFS to find if there is a path from the position of the agent to the new position

#    1
# 1  p  1
#    1

def find_targets(free_space, start, targets, logger=None):
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

def find_next_state(self, game_state):
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
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
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

