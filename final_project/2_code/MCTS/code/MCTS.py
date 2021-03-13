import math
import time
import itertools
import random
from copy import deepcopy
from MCTSinfor import *

class MCTS_Node(object):
    def __init__(self, parent, prob):
        self.parent = parent
        self.children = dict()
        self.visits = 0
        self.prob = prob
        self.Q_value = 0
        self.U_value = 0

    # expand child node
    def expand(self, action_p):
        for action, prob in action_p:
            if action not in self.children:
                self.children[action] = MCTS_Node(self, prob)

    # adjusted UCB method(considering the effect of visited times)
    def UCB_value(self, para):
        self.U_value = (para * self.prob * math.sqrt(self.parent.visits) / (1 + self.visits))
        return self.Q_value + self.U_value

    # adjusted UCB method to select children
    def select(self, para):
        result = max(self.children.items(), key=lambda x: x[1].UCB_value(para))
        return result

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False

    # Back propagation
    def back_update(self, value):
        if self.parent:
            self.parent.back_update(-value)
        self.visits += 1
        self.Q_value = self.Q_value + (value - self.Q_value) / self.visits


class Get(object):
    def __init__(self, depth=1, para=5):
        self.method = MCTS(depth, para)

    def action(self, board, time_limit):
        global endtime
        endtime = time_limit if time_limit != -1 else -1
        action = self.method.get_action(board)
        return action


class MCTS(object):
    def __init__(self, depth_limit=5, para=5):
        self.root = MCTS_Node(None, 1)
        self.para = para
        self.depth = depth_limit

    # simulate the Gomoku process
    def simulate(self, state):
        # simulation stage
        limit = 50
        board, player = state
        for depth in range(limit):
            if time.time() > endtime:
                return 0
            x, y = In_simulation((board, player))
            board[x][y] = player
            end = self.check_win(board, x, y, player)
            if end is True:
                return player
            player = 3 - player  # switch player
        return 0

    def rollout(self, state, simulation_time=3):
        board, player = state
        node = self.root

        # selection: find out the leaf node to be expanded
        end = False
        while not node.is_leaf():
            (x, y), node = node.select(self.para)
            board[x][y] = player
            player = 1 if player == 2 else 2    # switch player
            end = self.check_win(board, x, y, player)

        if end is False:
            action_prob = evaluation((board, player))
            node.expand(action_prob)
            # simulation
            opponent = 3 - player   # switch player
            for i in range(simulation_time):
                for action, _ in action_prob:
                    x, y = action
                    board_copy = deepcopy(board)
                    board_copy[x][y] = player
                    winner = self.simulate((board_copy, opponent))  # 0 for tie
                    leaf_value = -1 if winner == opponent else winner
                    node.children[action].back_update(leaf_value)
        elif end is True:
            node.back_update(1)
        else:
            node.back_update(0)

    def get_action(self, board):

        action = ifwin_action1(board, 1)
        if action is not None:
            return action
        action = ifwin_action1(board, 2)
        if action is not None:
            return action
        action = ifwin_action2(board, 1)
        if action is not None:
            return action
        action = iflose_action2(board, 2)
        if action is not None:
            return action
        action = ifwin_action3(board, 1)
        if action is not None:
            return action
        action = ifwin_action3(board, 2)
        if action is not None:
            return action

        if endtime == -1:
            actions = evaluation((board, 1))
            return max(actions, key=lambda x: x[1])[0]

        for n in range(self.depth):
            state_copy = deepcopy((board, 1))  # we are player 1
            self.rollout(state_copy)
            if time.time() > endtime:
                break
        return max(self.root.children.items(), key=lambda x: x[1].Q_value)[0]

    def brain_update(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            self.root = MCTS_Node(None, 1.0)

    def check_win(self, board, x, y, player):
        boardLength = len(board)
        # column
        d_start = max(-1 * x, -4)
        d_end = min(boardLength - x - 5, 0)
        for d in range(d_start, d_end + 1):
            pieces = [board[x + d + k][y] for k in range(5)]
            if pieces == [player] * 5:
                return True
        # row
        d_start = max(-1 * y, -4)
        d_end = min(boardLength - y - 5, 0)
        for d in range(d_start, d_end + 1):
            if board[x][y + d:y + d + 5] == [player] * 5:
                return True
        # diagonal
        d_start = max(-1 * x, -1 * y, -4)
        d_end = min(boardLength - x - 5, boardLength - y - 5, 0)
        for d in range(d_start, d_end + 1):
            pieces = [board[x + d + k][y + d + k] for k in range(5)]
            if pieces == [player] * 5:
                return True
        d_start = max(-1 * x, y - boardLength + 1, -4)
        d_end = min(boardLength - x - 5, y - 5, 0)
        for d in range(d_start, d_end + 1):
            pieces = [board[x + d + k][y - d - k] for k in range(5)]
            if pieces == [player] * 5:
                return True
        # tie (-1) or not terminal (False)
        for row in board:
            if 0 in row:
                return False
        return -1

# Return a list of the best n substates for current player
def evaluation(state):
    board, player = state
    moved = []
    for i in range(20):
        for j in range(20):
            if board[i][j] > 0:
                moved.append((i, j))

    substate = []
    adjacent = find_adj_2(moved)  # get the adjacent of the moved

    for (x, y) in adjacent:
        if board[x][y] > 0:
            continue
        else:
            board[x][y] = player
            score = board_evaluation(board, player)
            board[x][y] = 0
            # sum_score += score
            substate.append(((x, y), score))

    # choose the 3 sub-state with the highest scores
    sub = sorted(substate, key=lambda x: x[1], reverse=True)[:2]

    # normalization
    sub_norm = []
    max_score = sub[0][1]
    min_score = sub[-1][1]

    if max_score != min_score:
        for s in sub:
            sub_norm.append((s[0], (s[1] - min_score + 5) / (max_score - min_score) * 0.9))
    else:
        for s in sub:
            sub_norm.append((s[0], 0.9))

    return tuple(sub_norm)


# To get an action
def In_simulation(state):
    board, player = state

    # Test whether having kill action for player and opponent
    action = ifwin_action1(board, player)
    if action is not None:
        return action
    action = ifwin_action1(board, 3 - player)
    if action is not None:
        return action

    action = ifwin_action2(board, player)
    if action is not None:
        return action
    action = iflose_action2(board, 3 - player)
    if action is not None:
        return action

    action = ifwin_action3(board, player)
    if action is not None:
        return action
    action = ifwin_action3(board, 3 - player)
    if action is not None:
        return action

    moved = []
    for x, y in itertools.product(range(20), range(20)):
        if board[x][y] > 0:
            moved.append((x, y))
    adjacent = find_adj(moved)
    actions = list(set(adjacent) - set(moved))
    return random.choice(actions)
