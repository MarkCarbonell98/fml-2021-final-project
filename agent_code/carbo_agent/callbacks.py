import os
import pickle
import random
import sys
sys.path.append("agent_code/carbo_agent/")
from utils import *

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

t = 10
t0 = 4

def setup(self):
    pass

def act(self, game_state: dict) -> str:
    game_state = preprocess_game_state(game_state)
    action = decide(game_state, t, t0)
    return action

def preprocess_game_state(game_state: dict) -> dict:
    game_state['field'] = game_state['field'] + 1 # 0 = stone walls, 1 = free tile, 2 = crates
    return game_state
