import re
from collections import Counter

class_dict = {("WIN", (), ()): "11111",
                  ("H4", (0, 5), ()): "011110",
                  ("C4", (0), (-1)): r"01111($|2)",
                  ("C4", (5), (-1)): r"(^|2)11110",
                  ("T4", (0, 2, 6), ()): "10111",
                  ("T4", (0, 4, 6), ()): "11101",
                  ("T4", (0, 3, 6), ()): "11011",
                  ("H3", (0, 4), ()): "01110",
                  ("TH3", (0, 2, 5), ()): "010110",
                  ("TH3", (0, 3, 5), ()): "011010",
                  ("M3", (0, 1), (-1)): r"00111(2|$)",
                  ("M3", (4, 5), (-1)): r"(2|^)11100",
                  ("M3", (0, 2), (-1)): r"01011(2|$)",
                  ("M3", (3, 5), (-1)): r"(2|^)11010",
                  ("M3", (0, 3), (-1)): r"01101(2|$)",
                  ("M3", (2, 5), (-1)): r"(2|^)10110",
                  ("M3", (1, 2), ()): "10011",
                  ("M3", (2, 3), ()): "11001",
                  ("M3", (1, 3), ()): "10101",
                  ("M3", (1, 4), (-1, 6)): r"(2|^)011102",
                  ("M3", (1, 4), (0, -1)): r"201110(2|$)",
                  ("H2", (0, 1, 4), ()): "00110",
                  ("H2", (0, 3, 4), ()): "01100",
                  ("H2", (0, 2, 4), ()): "01010",
                  ("H2", (0, 2, 3, 5), ()): "010010",
                  ("M2", (0, 1, 2), (-1)): r"00011(2|$)",
                  ("M2", (3, 4, 5), (-1)): r"(2|^)11000",
                  ("M2", (0, 1, 3), (-1)): r"00101(2|$)",
                  ("M2", (2, 4, 5), (-1)): r"(2|^)10100",
                  ("M2", (0, 2, 3), (-1)): r"01001(2|$)",
                  ("M2", (2, 3, 5), (-1)): r"(2|^)10010",
                  ("M2", (1, 2, 3), (-1)): "10001",
                  ("M2", (1, 3, 5), (0, -1)): r"201010(2|$)",
                  ("M2", (1, 3, 5), (6)): r"^010102",
                  ("M2", (1, 4, 5), (6)): r"^011002",
                  ("M2", (1, 4, 5), (0, -1)): r"201100(2|$)",
                  ("M2", (1, 2, 5), (0, -1)): r"200110(2|$)",
                  ("M2", (1, 2, 5), (-1, 6)): r"(^|2)001102",
                  ("S4", (), (5)): r"(2|^)1111(2|$)",
                  ("S3", (), (0, 4)): "(2|^)111(2|$)",
                  ("S2", (), (0, 3)): "(2|^)11(2|$)",
                  }


def is_special_class(array, color):
    """
    judge whether the several chess given in the list form a special class
    """
    # add judgement here. Details in 'http://zjh776.iteye.com/blog/1979748'
    # array = copy.deepcopy(array)
    def _black_color(array):
        height, width = len(array), len(array[0])
        for i in range(height):
            for j in range(width):
                array[i][j] = (3 - array[i][j]) % 3
        return array

    if color == 2:  # 如果表明是对手(对手的数值为2)，则将棋盘翻转
        array = _black_color(array)

    height, width = len(array), len(array[0])
    class_counter = Counter()       # 专门用来计数的字典

    # scan by row
    for row_idx, row in enumerate(array):
        list_str = "".join(map(str, row))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by col
    for col_idx in range(width):
        col = [a[col_idx] for a in array]
        list_str = "".join(map(str, col))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by diag_1, from TL to BR
    for dist in range(-width + 1, height):
        row_ini, col_ini = (0, -dist) if dist < 0 else (dist, 0)
        diag = [array[i][j] for i in range(
            row_ini, height) for j in range(col_ini, width) if i - j == dist]
        list_str = "".join(map(str, diag))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by diag_2, from BL to TR
    for dist in range(0, width + height - 1):
        row_ini, col_ini = (dist, 0) if dist < height else (
            height - 1, dist - height + 1)
        diag = [array[i][j] for i in range(
            row_ini, -1, -1) for j in range(col_ini, width) if i + j == dist]
        list_str = "".join(map(str, diag))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    return class_counter


def class_to_score():
    """
    define the reward of some specific class of chess
    """
    score_map = {"WIN": 9999999999,  # 连五
                 "H4":  1000000,          # 活四
                 "C4":  1000,             # 冲四
                 "T4":  900,             # 跳四
                 # "C4": 1500,  # 冲四
                 # "T4": 1300,    # 跳四
                 "H3":  1000,             # 活三
                 "TH3": 800,            # 跳活三
                 "M3": 15,              # 眠三
                 "H2": 20,              # 活二
                 "M2": 5,               # 眠二
                 "SH2": 40,              # 双活二
                 "S4": -5,
                 "S3": -5,
                 "S2": -5
                 }

    return score_map


def extend_board(player, board):
    """
    add an edge for the board
    since the ege play the same role as the opponent for the player
    we set the ege as the opponent ( 3 - player )
    """
    k = len(board)  # the size of the board
    new_board = [[board[x-1][y-1] if 0 < x < k + 1 and 0 < y < k + 1 else 3 \
                  for x in range(k + 2)] for y in range(k + 2)]
    return new_board


def board_evaluation(board, player):
    score = 0

    brain_board = extend_board(board=board, player=player)
    for a_class, num in is_special_class(brain_board, 1).items():
        score = score + class_to_score()[a_class] * num

    opp = 3 - player
    oppo_board = extend_board(board=board, player=opp)
    for a_class, num in is_special_class(oppo_board, 2).items():
        score = score - class_to_score()[a_class] * num

    return score


def find_adj(moved):
    """
    find the neighbors of the moved
    """
    adjacent = set()
    width = 20
    height = 20

    for (h, w) in moved:
        if h < width - 1:
            adjacent.add((h+1, w))  # right
        if h > 0:
            adjacent.add((h-1, w))  # left
        if w < height - 1:
            adjacent.add((h, w+1))  # upper
        if w > 0:
            adjacent.add((h, w-1))  # lower
        if w < width - 1 and h < height - 1:
            adjacent.add((h+1, w+1))  # upper right
        if h > 0 and w < height - 1:
            adjacent.add((h-1, w+1))  # upper left
        if h < width - 1 and w > 0:
            adjacent.add((h+1, w-1))  # lower right
        if w > 0 and h > 0:
            adjacent.add((h-1, w-1))  # lower left

    adjacent = list(set(adjacent) - set(moved))
    return adjacent


def find_adj_2(moved):
    """
    find the neighbors of the moved
    """
    adjacent = set()
    width = 20
    height = 20

    for (h, w) in moved:
        if h < width - 1:
            adjacent.add((h+1, w))  # right
        if h > 0:
            adjacent.add((h-1, w))  # left
        if w < height - 1:
            adjacent.add((h, w+1))  # upper
        if w > 0:
            adjacent.add((h, w-1))  # lower
        if w < width - 1 and h < height - 1:
            adjacent.add((h+1, w+1))  # upper right
        if h > 0 and w < height - 1:
            adjacent.add((h-1, w+1))  # upper left
        if h < width - 1 and w > 0:
            adjacent.add((h+1, w-1))  # lower right
        if w > 0 and h > 0:
            adjacent.add((h-1, w-1))  # lower left

        # adj's adj
        if h < width - 2:
            adjacent.add((h+2, w))  # right
        if h-1 > 0:
            adjacent.add((h-2, w))  # left
        if w < height - 2:
            adjacent.add((h, w+2))  # upper
        if w > 1:
            adjacent.add((h, w-2))  # lower
        if w < width - 2 and h < height - 2:
            adjacent.add((h+2, w+2))  # upper right
        if h > 1 and w < height - 2:
            adjacent.add((h-2, w+2))  # upper left
        if h < width - 2 and w > 1:
            adjacent.add((h+2, w-2))  # lower right
        if w > 1 and h > 1:
            adjacent.add((h-2, w-2))  # lower left

    adjacent = list(set(adjacent) - set(moved))
    return adjacent


def ifwin_action1(board, player):
    # check if having four-in-a-row
    boardLength = len(board)
    # column
    for x in range(boardLength-4):
        for y in range(boardLength):
            pieces = tuple(board[x+d][y] for d in range(5))
            if pieces.count(player) == 4 and 0 in pieces:
                d = pieces.index(0)
                return x + d, y
    # row
    for x in range(boardLength):
        for y in range(boardLength-4):
            pieces = tuple(board[x][y+d] for d in range(5))
            if pieces.count(player) == 4 and 0 in pieces:
                d = pieces.index(0)
                return x, y + d
    # positive diagonal
    for x in range(boardLength-4):
        for y in range(boardLength-4):
            pieces = tuple(board[x+d][y+d] for d in range(5))
            if pieces.count(player) == 4 and 0 in pieces:
                d = pieces.index(0)
                return x + d, y + d
    # oblique diagonal
    for x in range(boardLength-4):
        for y in range(4, boardLength):
            pieces = tuple(board[x+d][y-d] for d in range(5))
            if pieces.count(player) == 4 and 0 in pieces:
                d = pieces.index(0)
                return x + d, y - d
    return None


def ifwin_action2(board, player):
    # check if having three-in-a-row
    boardLength = len(board)
    # column
    for x in range(boardLength-5):
        for y in range(boardLength):
            pieces = tuple(board[x+d][y] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                return x + d, y
    # row
    for x in range(boardLength):
        for y in range(boardLength-5):
            pieces = tuple(board[x][y+d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                return x, y + d
    # positive diagonal
    for x in range(boardLength-5):
        for y in range(boardLength-5):
            pieces = tuple(board[x+d][y+d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                return x + d, y + d
    # oblique diagonal
    for x in range(boardLength-5):
        for y in range(5, boardLength):
            pieces = tuple(board[x+d][y-d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                return x + d, y - d
    return None


def iflose_action2(board, player):
    # check if having three-in-a-row
    boardLength = len(board)
    # column
    for x in range(boardLength-5):
        for y in range(boardLength):
            pieces = tuple(board[x+d][y] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                actions = [(x, y), (x + 4, y)] if d == 4 else [(x + 1, y), (x + 5, y)] if d == 1 \
                           else [(x, y), (x + d, y), (x + 5, y)]
                max_s = float("-inf")
                max_act = None
                for act_x, act_y in actions:
                    board[act_x][act_y] = player
                    score = board_evaluation(board, player)
                    board[act_x][act_y] = 0
                    if score > max_s:
                        max_s = score
                        max_act = (act_x, act_y)
                return max_act
    # row
    for x in range(boardLength):
        for y in range(boardLength-5):
            pieces = tuple(board[x][y+d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                actions = [(x, y), (x, y + 4)] if d == 4 else [(x, y + 1), (x, y + 5)] if d == 1 \
                    else [(x, y), (x, y + d), (x, y + 5)]
                max_s = float("-inf")
                max_act = None
                for act_x, act_y in actions:
                    board[act_x][act_y] = player
                    score = board_evaluation(board, player)
                    board[act_x][act_y] = 0
                    if score > max_s:
                        max_s = score
                        max_act = (act_x, act_y)
                return max_act
    # positive diagonal
    for x in range(boardLength-5):
        for y in range(boardLength-5):
            pieces = tuple(board[x+d][y+d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                actions = [(x, y), (x + 4, y + 4)] if d == 4 else [(x + 1, y + 1), (x + 5, y + 5)] if d == 1 \
                    else [(x, y), (x + d, y + d), (x + 5, y + 5)]
                max_s = float("-inf")
                max_act = None
                for act_x, act_y in actions:
                    board[act_x][act_y] = player
                    score = board_evaluation(board, player)
                    board[act_x][act_y] = 0
                    if score > max_s:
                        max_s = score
                        max_act = (act_x, act_y)
                return max_act
    # oblique diagonal
    for x in range(boardLength-5):
        for y in range(5, boardLength):
            pieces = tuple(board[x+d][y-d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 3 and 0 in pieces[1:5]:
                d = pieces.index(0, 1, -1)
                actions = [(x, y), (x + 4, y - 4)] if d == 4 else [(x + 1, y - 1), (x + 5, y - 5)] if d == 1 \
                    else [(x, y), (x + d, y - d), (x + 5, y - 5)]
                max_s = float("-inf")
                max_act = None
                for act_x, act_y in actions:
                    board[act_x][act_y] = player
                    score = board_evaluation(board, player)
                    board[act_x][act_y] = 0
                    if score > max_s:
                        max_s = score
                        max_act = (act_x, act_y)
                return max_act
    return None


def ifwin_action3(board, player):
    # check if having double-four, one-four-one-three and double-three
    boardLength = len(board)
    # possible placement to achieve four
    # column
    col4 = set()
    for x in range(boardLength - 4):
        for y in range(boardLength):
            pieces = tuple(board[x + d][y] for d in range(5))
            if pieces.count(0) == 2 and pieces.count(player) == 3:
                for d in range(5):
                    if pieces[d] == 0:
                        col4.add((x + d, y))
    # row
    row4 = set()
    for x in range(boardLength):
        for y in range(boardLength - 4):
            pieces = tuple(board[x][y + d] for d in range(5))
            if pieces.count(0) == 2 and pieces.count(player) == 3:
                for d in range(5):
                    if pieces[d] == 0:
                        row4.add((x, y + d))
    # positive diagonal
    pos4 = set()
    for x in range(boardLength - 4):
        for y in range(boardLength - 4):
            pieces = tuple(board[x + d][y + d] for d in range(5))
            if pieces.count(0) == 2 and pieces.count(player) == 3:
                for d in range(5):
                    if pieces[d] == 0:
                        pos4.add((x + d, y + d))
    # oblique diagonal
    ob4 = set()
    for x in range(boardLength - 4):
        for y in range(4, boardLength):
            pieces = tuple(board[x + d][y - d] for d in range(5))
            if pieces.count(0) == 2 and pieces.count(player) == 3:
                for d in range(5):
                    if pieces[d] == 0:
                        ob4.add((x + d, y - d))

    # possible placement to achieve three
    # column
    col3 = set()
    for x in range(boardLength - 5):
        for y in range(boardLength):
            pieces = tuple(board[x + d][y] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 2 and pieces[1:5].count(0) == 2:
                for d in range(1, 5):
                    if pieces[d] == 0:
                        col3.add((x + d, y))
    # row
    row3 = set()
    for x in range(boardLength):
        for y in range(boardLength - 5):
            pieces = tuple(board[x][y + d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 2 and pieces[1:5].count(0) == 2:
                for d in range(1, 5):
                    if pieces[d] == 0:
                        row3.add((x, y + d))
    # positive diagonal
    pos3 = set()
    for x in range(boardLength - 5):
        for y in range(boardLength - 5):
            pieces = tuple(board[x + d][y + d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 2 and pieces[1:5].count(0) == 2:
                for d in range(1, 5):
                    if pieces[d] == 0:
                        pos3.add((x + d, y + d))
    # oblique diagonal
    ob3 = set()
    for x in range(boardLength - 5):
        for y in range(5, boardLength):
            pieces = tuple(board[x + d][y - d] for d in range(6))
            if pieces[0] == 0 and pieces[5] == 0 and pieces[1:5].count(player) == 2 and pieces[1:5].count(0) == 2:
                for d in range(1, 5):
                    if pieces[d] == 0:
                        ob3.add((x + d, y - d))

    sets = [col4, row4, pos4, ob4, col3, row3, pos3, ob3]
    for i in range(8):
        for j in range(i+1, 8):
            if j != i+4:
                intersect = sets[i] & sets[j]
                if len(intersect) != 0:
                    return list(intersect)[0]
