import pickle
import os
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 21 # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
alpha = 0.5
gamma = 0.6
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
    self.states = np.array([])
    self.coins_collected = 0
    self.steps = 0
    q_values = np.ones((2*2*2*2*11*11*11*11 + 2*2*2*2*11*11 + 2*2*2*2 + 1, 6))
    q_values[:, 5] = -40000
    q_values[:, 4] = -40000
    if self.train and os.path.isfile("q_values.npy"):
        q_values = np.load("q_values.npy")
        print("File was loaded")
    self.q_values = q_values
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
    if self_action != None:
        #self.states  = np.append(state_to_features(old_game_state))
        #if len(self.states)==5:
            #values, counts = np.unique(self.states, return_counts=True)

        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
        #print(events)
        if state_to_features(old_game_state) == state_to_features(new_game_state) and self.transitions:
            print("waited")
            state, action, next_state, reward = self.transitions.pop()
            reward = reward - 20
            if state != None or next_state != None or action !=None:
                if action =="UP":
                    action = 0
                if action =="RIGHT":
                    action =1
                if action == "DOWN":
                    action = 2
                if action == "LEFT":
                    action =3
                if action == "WAIT":
                    action = 4
                if action =="BOMB":
                    action =5
                next_max = np.max(self.q_values[next_state])
                old_value = self.q_values[state , action]
                if type(old_value)== type(np.float64(0.3)):
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    print(old_value)
                    print(new_value)
                    self.q_values[state, action] = new_value
            np.save("q_values.npy", self.q_values)
            self.coins_collected = 0
            self.steps = 0

        if (self.coins_collected % 2==0 and self.coins_collected>0):
            while self.transitions:
                state, action, next_state, reward = self.transitions.pop()
                if state != None or next_state != None or action !=None:
                    if action =="UP":
                        action = 0
                    if action =="RIGHT":
                        action =1
                    if action == "DOWN":
                        action = 2
                    if action == "LEFT":
                        action =3
                    if action == "WAIT":
                        action = 4
                    if action =="BOMB":
                        action =5
                    next_max = np.max(self.q_values[next_state])
                    old_value = self.q_values[state , action]
                    if type(old_value)== type(np.float64(0.3)):
                        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                        print(old_value)
                        print(new_value)
                        self.q_values[state, action] = new_value
            self.transitions.clear()
            print("deque was cleared")
            np.save("q_values.npy", self.q_values)
            print("2 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0
        numpy_events = np.array(events)
        if np.any(np.isin(numpy_events, 'INVALID_ACTION')) and self.transitions:
            last_transition = self.transitions.pop()
            state, action, next_state, reward = last_transition
            if state != None or next_state != None or action !=None:
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
                next_max = np.max(self.q_values[next_state])
                old_value = self.q_values[state, action]
                if type(old_value)== 'numpy.float64':
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    print(old_value)
                    print(new_value)
                    self.q_values[state, action] = new_value
                np.save("q_values.npy", self.q_values)
                print("An invalid move was made and the file was saved")
                self.steps = self.steps - 1
            if state == None:
                print("state = None")
            if next_state == None:
                print("new_state = None")
        if np.any(np.isin(numpy_events, 'COIN_COLLECTED')):
            print("coin was collected")
            self.coins_collected = self.coins_collected + 1
        # Idea: Add your own events to hand out rewards
        if ...:
            events.append(PLACEHOLDER_EVENT)
        # state_to_features is defined in callbacks.py
        self.steps = self.steps + 1
        if self.steps==5 and self.coins_collected ==1:
            while self.transitions:
                state, action, next_state, reward = self.transitions.pop()
                if state != None or next_state != None or action !=None:
                    if action =="UP":
                        action = 0
                    if action =="RIGHT":
                        action =1
                    if action == "DOWN":
                        action = 2
                    if action == "LEFT":
                        action =3
                    if action == "WAIT":
                        action = 4
                    if action =="BOMB":
                        action =5
                    next_max = np.max(self.q_values[next_state])
                    old_value = self.q_values[state,action]
                    if type(old_value)== type(np.float64(0.3)):
                        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                        print(old_value)
                        print(new_value)
                        self.q_values[state, action] = new_value
            self.transitions.clear()
            print("deque was cleared")
            np.save("q_values.npy", self.q_values)
            print("1 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0
        if self.steps == 5 and self.coins_collected == 0:
            state, action, next_state, reward = self.transitions.pop()
            reward = reward - 20
            if state != None or next_state != None or action !=None:
                while self.transitions:
                    if action =="UP":
                        action = 0
                    if action =="RIGHT":
                        action =1
                    if action == "DOWN":
                        action = 2
                    if action == "LEFT":
                        action =3
                    if action == "WAIT":
                        action = 4
                    if action =="BOMB":
                        action =5
                    next_max = np.max(self.q_values[next_state])
                    old_value = self.q_values[state,action]
                    print(old_value)
                    if type(old_value)== type(np.float64(0.3)):
                        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                        self.q_values[state, action] = new_value
                    state, action, next_state, reward = self.transitions.pop()
                    reward = reward - 20
            self.transitions.clear()
            print("deque was cleared")
            np.save("q_values.npy", self.q_values)
            print("0 Coins were collected and the file was saved")
            self.coins_collected = 0
            self.steps = 0
    if self_action == None:
        print("Self_action = None")
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    end_state = state_to_features(last_game_state)
    action = last_action
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
    end_action = action
    end_reward = reward_from_events(self, events)
    if (end_state != None or end_action != None) and self.transitions:
        state, action, vorletzter_state, reward = self.transitions.pop()
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
        next_max = np.max(self.q_values[end_state])
        old_value = self.q_values[vorletzter_state, end_action]
        if type(old_value) == type(np.float64(0.3)):
            new_value = (1 - alpha) * old_value + alpha * (end_reward + gamma * next_max)
            self.q_values[vorletzter_state, action] = new_value
        next_state = vorletzter_state
        next_max = np.max(self.q_values[next_state])
        old_value = self.q_values[state, action]
        if type(old_value) == type(np.float64(0.3)):
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            self.q_values[state, action] = new_value
        if state != None or next_state != None or action != None:
            while self.transitions:
                state, action, next_state, reward = self.transitions.pop()
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
                next_max = np.max(self.q_values[next_state])
                old_value = self.q_values[state, action]
                if type(old_value) == type(np.float64(0.3)):
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    print(old_value)
                    print(new_value)
                    self.q_values[state, action] = new_value
    self.transitions.clear()
    print("deque was cleared")
    np.save("q_values.npy", self.q_values)
    print("Game Ended")
    # Store the model
    #with open("my-saved-model.pt", "wb") as file:
     #   pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.MOVED_DOWN: -.1,
        e.WAITED: -.1,
        e.KILLED_SELF: -100,
        e.INVALID_ACTION:-5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

class MyModel:
    def __init__(self, q_values):
        self.q = q_values
    def propose_action(game_state):
        features = state_to_features(game_state)