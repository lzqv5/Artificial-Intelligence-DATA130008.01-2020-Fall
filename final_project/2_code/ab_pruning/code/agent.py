import evaluation

state = {'US':0,'OP':1,"EMPTY":2}

def offsets(x,y,direction): # 返回起点和移动方向
    if direction == 0:  # 一共四个方向
        return ((x,y+4),0,-1)
    if direction == 1:
        return ((x-4,y+4),1,-1)
    if direction == 2:
        return ((x-4,y),1,0)
    if direction == 3:
        return ((x-4,y-4),1,1)

def opponent(p):
    if p == state["US"]:
        return state["OP"]
    elif p == state["OP"]:
        return state["US"]
    raise ValueError("Invalid Player.")

def pattern2points(patter_controller,pattern_opponent): # 将特定的模式转换为对应的分数
    return evaluation.POINTS[patter_controller][pattern_opponent]

def points2priority(point1,point2,point3,point4):   # 将四个方向的分数转化为该点的priority
    return evaluation.PRIORITY[point1][point2][point3][point4]

class Unit:     # 对单个落子位点的抽象 - 存储了相关位点的模式、分数、优先级等信息
    # 棋盘上的每个位置
    def __init__(self,x=0,y=0):
        self.type = state["EMPTY"]
        self.x = x
        self.y = y
        self.neighbor = 0 # 相邻单元已落子的个数
        self.value = [0,0]
        # self.value = 0
        self.patterns = [[0,0] for i in range(4)] # store 8 binary numbers which denote the patterns in this unit.
        self.points = [[0,0] for i in range(4)] # 4 directions 2 players
        self.priority = [0,0]

    def updatePoints(self,direction):
        self.points[direction][0] = pattern2points(self.patterns[direction][0]
                                                   ,self.patterns[direction][1])
        self.points[direction][1] = pattern2points(self.patterns[direction][1]
                                                   ,self.patterns[direction][0])

    def updataPriority(self):
        self.priority[0] = points2priority(self.points[0][0],self.points[1][0],self.points[2][0],self.points[3][0])
        self.priority[1] = points2priority(self.points[0][1],self.points[1][1],self.points[2][1],self.points[3][1])

    def eval(self): # 对单个落子点评分 - 衡量单个落子位置的重要性 - # point evaluation
        p0 = 0
        p1 = 0
        for direction in range(4):
            p0 += evaluation.ReGrade[self.points[direction][0]]
            p1 += evaluation.ReGrade[self.points[direction][1]]
        self.value[0] = p0
        self.value[1] = p1

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __str__(self):  # for debug
        return str(((self.x,self.y),self.value))

    __repr__ = __str__  # for debug


class Board:
    # agent
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.board = [[Unit(x,y) for y in range(height)] for x in range(width)]
        self.controller = state["US"]
        self.opponent = state["OP"]
        # self.candidates = []
        self.moveCount = 0  # 落子数/下棋数
        self.record = []  # 落子位置对应的单元cell
        self.specialStates = [[0 for i in range(10)] for j in range(2)]  # 记录特殊模式
        self.MAXDEPTH = 6
        self.branchFactor = 10
        self.WIN = 50000

    def isLegal(self,x, y):
        return x >= 0 and y >= 0 and x < self.width and y < self.height

    def boardInitialization(self):  # 棋盘初始化
        for x in range(self.width):
            for y in range(self.height):
                for direction in range(4):
                    v = 0b00000001
                    count = 1
                    startPoints,dx,dy = offsets(x,y,direction)
                    xn,yn = startPoints[0],startPoints[1]
                    while v <= 128:
                        if not self.isLegal(xn,yn):
                            self.board[x][y].patterns[direction][0] ^= v
                            self.board[x][y].patterns[direction][1] ^= v
                        count += 1
                        v <<= 1
                        if count == 5: # 跳过中心点
                            xn += 2*dx
                            yn += 2*dy
                        else:
                            xn += dx
                            yn += dy
        for x in range(self.width): # 更新邻居的模式和分数
            for y in range(self.height):
                for direction in range(4):
                    self.board[x][y].updatePoints(direction)
                self.board[x][y].updataPriority()

    def setController(self,player): # 设置 agent 的控制者
        self.controller = player
        self.opponent = opponent(player)

    def move(self,x,y): # 下棋
        self.specialStates[0][self.board[x][y].priority[0]] -= 1
        self.specialStates[1][self.board[x][y].priority[1]] -= 1
        self.board[x][y].type = self.controller  # 表示这个位置已经被当前操控Agent的人占有了
        self.record.append(self.board[x][y])
        self.moveCount += 1

        for direction in range(4):
            startPoint,dx,dy = offsets(x,y,direction)
            xn,yn = x + 4*dx , y+4*dy
            count = 1
            v = 0b00000001
            while v <= 128:
                if self.isLegal(xn,yn):
                    self.board[xn][yn].patterns[direction][self.controller] ^= v
                    if self.board[xn][yn].type == state["EMPTY"]:
                        self.specialStates[0][self.board[xn][yn].priority[0]] -= 1
                        self.specialStates[1][self.board[xn][yn].priority[1]] -= 1
                        self.board[xn][yn].updatePoints(direction)
                        self.board[xn][yn].updataPriority()
                        self.specialStates[0][self.board[xn][yn].priority[0]] += 1
                        self.specialStates[1][self.board[xn][yn].priority[1]] += 1
                v <<= 1
                count += 1
                if count == 5:
                    xn -= 2*dx
                    yn -= 2*dy
                else:
                    xn -= dx
                    yn -= dy

        for i in [-1,0,1]:  # 挑选一阶邻居
            for j in [-1,0,1]:
                if i == 0 and j == 0:
                    continue
                try:
                    self.board[x+i][y+j].neighbor += 1
                except IndexError as e:
                    continue
        # 交换选手 - 轮次更换
        self.setController(opponent(self.controller))

    def withdraw(self): # 悔棋
        self.moveCount -= 1
        reset = self.record.pop()
        x,y = reset.x, reset.y
        self.board[x][y].type = state["EMPTY"]
        for direction in range(4):
            self.board[x][y].updatePoints(direction)
        self.board[x][y].updataPriority()
        self.specialStates[0][self.board[x][y].priority[0]] += 1
        self.specialStates[1][self.board[x][y].priority[1]] += 1
        self.setController(opponent(self.controller))   # exchange players
        # 修改相邻棋子的模式
        for direction in range(4):
            startPoint, dx, dy = offsets(x, y, direction)
            xn, yn = x + 4 * dx, y + 4 * dy
            count = 1
            v = 0b00000001
            while v <= 128:
                if self.isLegal(xn, yn):
                    self.board[xn][yn].patterns[direction][self.controller] ^= v
                    if self.board[xn][yn].type == state["EMPTY"]:
                        self.specialStates[0][self.board[xn][yn].priority[0]] -= 1
                        self.specialStates[1][self.board[xn][yn].priority[1]] -= 1
                        self.board[xn][yn].updatePoints(direction)
                        self.board[xn][yn].updataPriority()
                        self.specialStates[0][self.board[xn][yn].priority[0]] += 1
                        self.specialStates[1][self.board[xn][yn].priority[1]] += 1
                v <<= 1
                count += 1
                if count == 5:
                    xn -= 2 * dx
                    yn -= 2 * dy
                else:
                    xn -= dx
                    yn -= dy

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                try:
                    self.board[x + i][y + j].neighbor -= 1
                except IndexError as e:
                        continue

    def boardEval(self):    # board evaluation
        score = {self.controller:0, self.opponent:0}
        for i in range(self.moveCount): # 遍历所有已落子位置，传入棋子的模式 - 根据传入模式相关变量的个数，有权加和得到分数
            unit = self.record[i]
            for direction in range(4):
                score[unit.type] += evaluation.SCORE[unit.patterns[direction][unit.type]][unit.patterns[direction][1-unit.type]]
        return score[self.controller] - score[self.opponent]


    def fixedResponse(self,player,prior,cands): # 寻找目标candidate
        i = 0   # 寻找具有目标优先级的successor
        while (self.board[cands[i].x][cands[i].y].priority[player] != prior):
            i += 1
        return [cands[i]]


    def getCandidates(self):    # generate candidates
        cands = []  # 产生备选落子位置
        n = 0
        for x in range(self.width):
            for y in range(self.height):
                if (self.board[x][y].type==state["EMPTY"]) and \
                        (self.board[x][y].neighbor or \
                         max(self.board[x][y].priority)>=evaluation.E): # 二阶且周围无棋子的邻居
                    self.board[x][y].eval()
                    cands.append(self.board[x][y])
                    n += 1
        if self.specialStates[self.controller][evaluation.A] > 0:
            return self.fixedResponse(self.controller,evaluation.A,cands),1
        if self.specialStates[self.opponent][evaluation.A] > 0:
            return self.fixedResponse(self.opponent,evaluation.A,cands),1
        if self.specialStates[self.controller][evaluation.B] > 0:
            return self.fixedResponse(self.controller,evaluation.B,cands),1
        if self.specialStates[self.opponent][evaluation.B] > 0:
            candsNew = []
            n = 0
            for cand in cands:
                x,y = cand.x,cand.y
                condition1 = self.board[x][y].priority[self.controller] >= evaluation.E
                condition2 = self.board[x][y].priority[self.opponent] >= evaluation.E
                if condition1 or condition2 :
                    candsNew.append(cand)
                    n += 1
            return candsNew,n
        return cands,n

    def killSearch(self):   # Threat Space Search - TSS
        if self.specialStates[self.controller][evaluation.A] >= 1 :
            return 1    # win
        if self.specialStates[self.opponent][evaluation.A] > 1: # more than one five-in-row
            return -1   # lose
        if self.specialStates[self.opponent][evaluation.A] == 1:
            cands,n = self.getCandidates()  # 迭代调用，进行推理
            for cand in cands:
                if self.board[cand.x][cand.y].priority[self.opponent] == evaluation.A:
                    self.move(cand.x,cand.y)
                    res = -self.killSearch()
                    self.withdraw()
                    return res
        if self.specialStates[self.controller][evaluation.B] >= 1:
            return 1
        if self.specialStates[self.controller][evaluation.C] >= 1:
            if self.specialStates[self.opponent][evaluation.B] == 0:
                cands,n = self.getCandidates()
                for cand in cands:
                    if self.board[cand.x][cand.y].priority[self.opponent] == evaluation.A:
                        self.move(cand.x, cand.y)
                        res = -self.killSearch()
                        self.withdraw()
                        return res
        return 0 # not sure. - 判断不了胜负


    def minimax(self,depth,alpha,beta): # minimax search
        res = self.killSearch() # TSS
        if res:
            if depth: # 非根节点
                if res>0:
                    return self.WIN,[0,0]
                else:
                    return -self.WIN,[0,0]
            if res>0:   # 根节点
                cands, n = self.getCandidates()
                for cand in cands:
                    if self.board[cand.x][cand.y].priority[self.controller] == evaluation.A:
                        return self.WIN+10,[cand.x,cand.y]

        if depth == self.MAXDEPTH: # 叶节点
            return self.boardEval(),(0,0)
        cands, n = self.getCandidates()
        if n==0:
            return 0,[0,0]
        cands.sort(key=lambda cand: -sum(cand.value)) # 降序
        # cands.sort(key=lambda cand: -cand.value)  # 降序
        cands = cands[0:min(n,self.branchFactor)]  # 每一层都视为MAX结点
        v = -float('inf')
        bestv = 0
        bestpos = [0,0]
        for cand in cands:
            self.move(cand.x,cand.y)
            m, _ = self.minimax(depth + 1, -beta, -alpha)
            self.withdraw()
            if -m > v:
                v = -m
                bestv = v
                bestpos[0] = cand.x
                bestpos[1] = cand.y
            alpha = max(alpha,v)
            if v >= beta: # pruning
                return bestv,bestpos
        return bestv,bestpos

    def reasoing(self):
        if self.moveCount == 0:
            return self.width//2, self.height//2
        _,best = self.minimax(0,float('-inf'),float('inf'))
        return best[0],best[1]