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

def build_map_graph(game_state: dict) -> list:
    pass

def a_star_path(s: tuple, t: tuple, game_state: dict) -> list:
    pass

def find_safe_place(game_state: dict) -> tuple:
    pass

def find_nearest_crate(game_state: dict) -> tuple:
    pass

def find_nearest_coin(game_state: dict) -> tuple:
    pass

def find_nearest_opponent(game_state: dict) -> tuple:
    pass



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
