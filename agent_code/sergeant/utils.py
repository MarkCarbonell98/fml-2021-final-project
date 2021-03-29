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


def write_dict_to_file(self):
    """
    JSON to structure data as a dictionary
    Args:
        self:
    """
    with open('q_table.json', 'w') as output:
        output.write(json.dumps(self.q_table))
        output.close()


def read_dict_from_file(self):
    """
    Read q_table dictionary from JSON
    Args:
        self:
    """
    with open('q_table.json', 'r') as q_table:
        self.q_table = json.load(q_table)
        q_table.close()


def training_radius(self):
    """
    Radius is set according to coins instead of crates for training with coins
    Args:
        self:
    """
    coins = np.array(self.game_state['coins'])
    agent = np.array([self.game_state['self'][3][0], self.game_state['self'][3][1]])
    self.radius = np.max(np.min(np.linalg.norm(agent - coins, axis=1)), 4 * np.sqrt(2))


def real_radius(self):
    """
    Calculate searching radius according to crates
    Args:
        self:
    """
    crates = np.transpose(np.array(np.where(self.game_state['field'] == 1)))
    agent = np.array([self.game_state['self'][3][0], self.game_state['self'][3][1]])
    norm = np.linalg.norm(agent - crates, axis=1)
    self.radius = np.max([np.min(norm), 4 * np.sqrt(2)])


def state_to_str(state):
    """
    Changes list with states to string
    Strings are used as keys for the q_table dictionary
    Args:
        state:

    Returns:

    """
    return ", ".join(state)


def dangerous_bombs(self, pos):
    """
    Identifies bombs that put agent in danger
    Args:
        self:
        pos: coordinates of tile in danger

    Returns: list of bombs that put agent in danger

    """
    # Bombs position
    bombs = np.array([x, y] for x, y, t in self.game_state['bombs'] if (x == pos[0][0] or y == pos[0][1]))

    if len(bombs):
        bombs_distance = np.linalg.norm(bombs - pos, axis=1)
        bombs = bombs[bombs_distance < 4]
        if len(bombs):
            # Distance from dangerous bombs
            dangerous_bombs = bombs_distance[bombs_distance < 4]
            # Determine direction of the bomb
            temp = np.sign(bombs - pos)

            indices = []
            for i, direction in enumerate(temp):
                # clear is a boolean variable to indicate if tile is clear
                temp_pos, clear = pos, False
                for _ in range(int(dangerous_bombs)):
                    temp_pos = np.array([temp_pos[0] + direction[0], temp_pos[1] + direction[1]])
                    if self.game_state['field'][temp_pos[0], temp_pos[1]] == -1:
                        # There is a bomb between wall and agent
                        clear = True
                        break
                if not clear:
                    # Agent is in danger
                    indices.append(i)
            return bombs[np.array(indices)] if len(indices) else []
    return []


def find_priority(self, current_state, cand_name):
    """
    Score among possible movements is computed.
    Score consists on a weighted sum of crate, coin, and enemy considering the squared relative distance from
    the checked position.
    Args:
        self:
        current_state: current built state
        cand_name: Conditional state (empty, danger1)

    Returns:

    """
    candidates = np.array(current_state)[:4]
    candidates = np.where(candidates == cand_name)[0]

    if not len(candidates):
        return

    # Calculate priority scores for candidate directions
    scores, best_indices = np.zeros(len(candidates)), []
    for i, index in enumerate(candidates):
        col, row = self.changes_point_in_dir[index]
        # Supposed position after movement
        temp_pos = np.array([self.curr_pos[0] + col, self.curr_pos[1] + row])

        if cand_name == 'danger':
            bombs_cord = dangerous_bombs(self, temp_pos)
            if len(bombs_cord):
                distance = np.linalg.norm(temp_pos - bombs_cord, axis=1)
                closest_bomb = bombs_cord[np.argmin(distance)]
                if min(distance) < np.linalg.norm(self.curr_pos - closest_bomb):
                    scores[i] = np.inf
                    continue
                else:
                    scores[i] += len(bombs_cord) * -20
                    bombs = np.array(self.game_state['bombs'])[:, [0, 1]]
                    distance = np.linalg.norm(temp_pos - bombs)
                    scores[i] += np.sum(-20 * (1 / distance))

        # Assign weight according to object
        for object_name in self.weights.keys():
            if object_name == 'field':
                objects = np.array(np.where(self.game_state['field'] == 1))
                objects = objects.transpose()

            elif object_name == 'coins':
                objects = np.array(self.game_state['coins'])
            else:
                enemies = np.array(self.game_state['others'])
                objects = enemies[:, [0, 1]].astype(int) if len(enemies) else []

            if len(objects):
                # Calculate distance between supposed position after move and objects
                distance = np.linalg.norm(temp_pos - objects, axis=1)
                indices = np.where(distance <= self.radius)[0]
                rel_distance = distance[indices]
                if len(rel_distance):
                    temp = np.ones(len(rel_distance)) * self.weights[object_name]
                    scores[i] += np.sum(temp / ((rel_distance + 1) ** 2))

    # Choose the best route according to the scores.
    if cand_name == 'danger':
        scores = np.array(scores)
        candidates = candidates[scores != np.inf]
        scores = scores[scores != np.inf]
        if len(scores) == 0:
            return
        best_indices = candidates[np.where(np.array(scores) == max(scores))[0]]
    else:
        best_indices = candidates[np.where(np.array(scores) == max(scores))[0]]

    j = np.random.choice(best_indices)
    current_state[j] = 'priority'


def danger_boolean(self, pos):
    """
    Determines if agent is in danger
    Args:
        self:
        pos: checked tilw

    Returns:

    """
    bombs = dangerous_bombs(self, pos)
    if len(bombs):
        bombs_dis = np.linalg.norm(bombs - pos, axis=1)
        return len(np.where(bombs_dis < 4)[0]) > 0
    return False


def clear_path(self, idx, curr_pos, counter=4):
    """
    For each possible direction (empty tiles), a path of 4 tiles is recalculated in order to escape from a bomb.
    This is done recursively.
    Args:
        agent:
        idx: direction (0: left, 1: up, 2: right, 3: down)
        curr_pos: current position of the agent
        counter: number of tiles left to check in path.

    Returns: boolean - 1 if the path is clear, 0 if not

    """
    try:
        dir_var = self.changes_point_in_dir[idx]
        new_pos = np.array([curr_pos[0] + dir_var[0], curr_pos[1] + dir_var[1]])
        if counter == 0:
            return True
        elif self.game_state['field'][new_pos[0], new_pos[1]] != 0:
            return False
        elif self.game_state['explosion_map'][new_pos[0], new_pos[1]] > 2:
            return False
        elif len(self.game_state["others"]):
            for enemy in self.game_state["others"]:
                if enemy[0] == new_pos[0] and enemy[1] == new_pos[1]:
                    return False

        # Search for an escape rout
        for direction in [0, 1, 2, 3]:
            temp_dir_var = self.changes_point_in_dir[direction]
            if np.abs(direction - idx) != 2 and direction != idx and self.game_state['field'][
                new_pos[0] + temp_dir_var[0], new_pos[1] + temp_dir_var[1]] == 0:
                return True

        # Next tile in path
        return clear_path(self, idx, new_pos, counter - 1)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Reward_update: ", e, exc_type, fname, exc_tb.tb_lineno)
        bul = 5


def find_state(self):
    """
    1st.: In each iteration, the state of each tile in all possible directions is determined and analyzed
    2nd.: Priority policy: If no imminent danger, best move will be calculated
    Policies: Grading of each empty direction by number of objects on field taking into consideration their
    respective weights and distance from the agent.
    Direction's value = \Sum_i object_i * weight(object_i) / distance_from_agent(object_i)

    Args:
        self:

    Returns:

    """
    options = [(-1, 0, 'left'), (0, -1, 'up'), (1, 0, 'right'), (0, 1, 'down'), (0, 0, 'self')]
    curr_state = []
    agent_coordinates = np.array([self.game_state['self'][3][0], self.game_state['self'][3][1]])
    self.curr_pos = agent_coordinates

    for col, row, direction in options:
        # Tile to be analyzed
        observed_coordinates = np.array([agent_coordinates[0] + col, agent_coordinates[1] + row])

        # Determine if tile is full or not
        if self.game_state['field'][observed_coordinates[0], observed_coordinates[1]] == -1:
            curr_state.append('wall')
            continue
        elif self.game_state['field'][observed_coordinates[0], observed_coordinates[1]] == 1:
            curr_state.append('crate')
            continue
        elif self.game_state['explosion_map'][observed_coordinates[0], observed_coordinates[1]] > 1:
            curr_state.append('wall')
            continue

        # Determine players on the field and get distance from agent
        players = np.array(self.game_state['others'])
        if len(players):
            players_dis = np.linalg.norm(players[:, [0, 1]].astype(int) - observed_coordinates, axis=1)
            if len(players[players_dis == 0]):
                curr_state.append('enemy')
                continue

        # Determine bombs on the field and get coordinates
        bombs = np.array([[x, y] for (x, y), t in self.game_state['bombs'] if
                          (x == observed_coordinates[0] or y == observed_coordinates[1])])

        if len(bombs):
            bombs_distance = np.linalg.norm(bombs - observed_coordinates, axis=1)
            if len(bombs[bombs_distance == 0]) > 0:
                curr_state.append('bomb')
                continue

            if danger_boolean(self, observed_coordinates):
                curr_state.append('danger')
                continue

        # Stay at the same place is an empty field
        if direction == 'self':
            curr_state.append('empty')
            continue

        # Determine coins on the field
        coins = np.array(self.game_state['coins'])
        if len(coins):
            coins = np.linalg.norm(coins - observed_coordinates, axis=1)
            if len(coins[coins == 0]) > 0:
                curr_state.append('coin')
                continue

        if len(players):
            players_dis = np.linalg.norm(players[:, [0, 1]].astype(int) - observed_coordinates, axis=1)
            if len(players[players_dis == 0]) > 0:
                curr_state.append('enemy')
                continue

        # Tile is empty
        curr_state.append("empty")

    # Check for danger
    if curr_state[4] == 'bomb':
        empty = np.where(np.array(curr_state)[:4] == 'danger')[0]
        for j in empty:
            if not clear_path(self, j, self.curr_pos):
                curr_state[j] = 'wall'

    if not ('enemy' in curr_state[:4] and self.game_state['self'][2] == 1) and 'coin' not in curr_state[:4]:
        # Avoid getting stuck by own bomb.
        if 'crate' in curr_state[:4] and self.game_state["self"][2]:
            empty = np.array(curr_state)[:4]
            empty = np.where(empty == 'empty')[0]
            for j in empty:
                if not clear_path(self, j, self.curr_pos):
                    curr_state[j] = 'wall'

        # Find priority of directions with empty tiles
        elif 'empty' in curr_state[:4]:
            find_priority(self, curr_state, 'empty')

        # If there is no empty tile around us, but there is danger, we should try and find out
        # If we still should move into it, another one, or wait in our place.
        elif 'danger' in curr_state[:4]:
            find_priority(self, curr_state, "danger")

    # Boolean indicating whether bomb action is possible
    curr_state.append(str(self.game_state["self"][2] == 0))

    # If state is not in q_table, add it with a list of 6 0's.
    string_state = state_to_str(curr_state)
    try:
        self.q_table[string_state]
    except Exception as ex:
        self.q_table[string_state] = list(np.zeros(6))

    return curr_state
