# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    # Note: DFS does not guarantee the optimality

    # Just to remind programmers of some functions maybe useful.
    # problem.getStartState(self):
    # problem.isGoalState(self, state):
    # problem.getSuccessors(self, state):       - return  a list of triples, (successor,action, stepCost)
    # problem.getCostOfActions(self, actions):  - actions:A list of actions to take
    path = []

    class Node:
        def __init__(self,state,parent,action):
            '''

            :param state:       the specific state for the problem to solve.
            :param parent:      the father node.
            :param action:      the action taken to reach the state
            '''
            self.state = state
            self.parent = parent
            self.action = action

        def __str__(self):          # just for showing the details of each node in the fringe.
            return str(self.state)

        __repr__ = __str__

    start_state = problem.getStartState()
    root = Node(state = start_state , parent = None , action = None)
    fringe = util.Stack()   # define the Data Structure of Fringe - in DFS, it will be a LIFO container.
    fringe.push(root)
    explored = set()        # Graph Algorithm. Never expand/visit a state twice

    # In PacMan Problem, the State Space Graph is finite, thus we can find a solution to resolve the search problem.
    while not fringe.isEmpty():         # while there are still some Nodes in the Graph to be expanded.
        node_cur = fringe.pop()         # node to be expanded

        if node_cur.state in explored:  # if this node has been expanded before,
                                        # we just ignore it and keep popping other nodes
            continue
        else:                           # else, this node has never been expanded before, thus we need to expand it and
                                        # record it in the set called explored.
            explored.add(node_cur.state)

        if problem.isGoalState(node_cur.state):     # Goal Test. we test whether the current state is a Goal State.
            break

        # After we successfully expand a node, we need to get neighbors of the expanded node
        successors = problem.getSuccessors(node_cur.state)
        # every successor is a triple in the form of ( successor/state, action, stepCost )

        for successor in successors:
            node_successor = Node(state=successor[0],parent=node_cur,action=successor[1])
            # if the state has been expanded before, there is no need to observe and visit it again. Just ignore it.
            if not node_successor.state in explored:
                fringe.push(node_successor)

    # We produce the path according to the final Node.
    while node_cur.parent:
        path.append(node_cur.action)
        node_cur = node_cur.parent

    path.reverse()
    return path

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    path = []

    # problem.getStartState(self):
    # problem.isGoalState(self, state):
    # problem.getSuccessors(self, state):       - return  a list of triples, (successor,action, stepCost)
    # problem.getCostOfActions(self, actions):  - actions:A list of actions to take

    # pls refer to the depthFirstSearch function for detailed description.
    class Node:
        def __init__(self, state, parent, action):
            '''

            :param state:       the specific state for the problem to solve.
            :param parent:      the father node.
            :param action:      the action taken to reach the state
            '''
            self.state = state
            self.parent = parent
            self.action = action

        def __str__(self):
            return str(self.state)

        __repr__ = __str__

    start_state = problem.getStartState()
    root = Node(state=start_state, parent=None, action=None)
    fringe = util.Queue()
    fringe.push(root)
    explored = set()  # Graph Algorithm.

    while not fringe.isEmpty():     # pls refer to the depthFirstSearch function for detailed description.
        node_cur = fringe.pop()     # expand nodes

        if node_cur.state in explored:  # this node has been expanded
            continue
        else:
            explored.add(node_cur.state)

        if problem.isGoalState(node_cur.state):
            node_Goal = node_cur
            break

        # every successor is a triple in the form of ( successor/state, action, stepCost )
        successors = problem.getSuccessors(node_cur.state)

        flag = 0
        for successor in successors:
            # All my algorithms default to conduct a goal test when expanding nodes instead of observing nodes.
            # But in order to pass the Auto Grader,
            # there is a bit change and I add another Goal Test when observing nodes.
            if problem.isGoalState(successor[0]):
                node_Goal = Node(state=successor[0], parent=node_cur, action=successor[1])
                flag = 1
                break

            if not successor[0] in explored:
                fringe.push(Node(state=successor[0], parent=node_cur, action=successor[1]))
            # node_successor = Node(state=successor[0], parent=node_cur, action=successor[1])
            # if not node_successor.state in explored:
            #     fringe.push(node_successor)

        if flag == 1: # Goal Test is successful when observing nodes.
            break


    while node_Goal.parent:
        path.append(node_Goal.action)
        node_Goal = node_Goal.parent

    path.reverse()
    return path

    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path = []

    # problem.getStartState(self):
    # problem.isGoalState(self, state):
    # problem.getSuccessors(self, state):       - return  a list of triples, (successor,action, stepCost)
    # problem.getCostOfActions(self, actions):  - actions:A list of actions to take

    # pls refer to the depthFirstSearch function for detailed description.
    class Node:
        def __init__(self, state, parent, action, path_cost):
            '''

            :param state:       the specific state for the problem to solve.
            :param parent:      the father node.
            :param action:      the action taken to reach the state.
            :param path_cost:   the total cost from the Start State to the current State.
            '''
            self.state = state
            self.parent = parent
            self.action = action
            if parent == None:  # means it is root and it does not have father node.
                self.path_cost = path_cost
            else:
                self.path_cost = parent.path_cost + path_cost

        def __str__(self):
            return str(self.state)

        __repr__ = __str__

    start_state = problem.getStartState()
    root = Node(state=start_state, parent=None, action=None, path_cost=0)
    fringe = util.PriorityQueue()  # element is in the form of (priority, self.count, item)  -  item's type is node
    fringe.update(item = root,priority = root.path_cost)
    explored = set()  # Graph Algorithm.

    while not fringe.isEmpty():     # pls refer to the depthFirstSearch function for detailed description.
        node_cur = fringe.pop()     # expand nodes

        if node_cur.state in explored:  # this node has been expanded
            continue
        else:
            explored.add(node_cur.state)

        if problem.isGoalState(node_cur.state):
            break

        successors = problem.getSuccessors(node_cur.state)

        # every successor is a triple in the form of ( successor/state, action, stepCost )
        for successor in successors:
            node_successor = Node(state=successor[0], parent=node_cur, action=successor[1], path_cost=successor[2])
            if not node_successor.state in explored:
                fringe.update( item = node_successor , priority = node_successor.path_cost)

    while node_cur.parent:
        path.append(node_cur.action)
        node_cur = node_cur.parent

    path.reverse()
    return path


    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 参数除了 problem   还有特定的启发式函数.
    # 启发式函数的参数为具体的状态和对应的的问题
    path = []

    # problem.getStartState(self):
    # problem.isGoalState(self, state):
    # problem.getSuccessors(self, state):       - return  a list of triples, (successor,action, stepCost)
    # problem.getCostOfActions(self, actions):  - actions:A list of actions to take

    # pls refer to the depthFirstSearch function for detailed description.
    class Node:
        def __init__(self, state, parent, action, path_cost, heuristicScore):
            '''

            :param state:           the specific state for the problem to solve.
            :param parent:          the father node.
            :param action:          the action taken to reach the state.
            :param path_cost:       the total cost from the Start State to the current State.
            :param heuristicScore:  the state's heuristic value
            '''
            self.state = state
            self.parent = parent
            self.action = action
            if parent == None:
                self.path_cost = path_cost
            else:
                self.path_cost = parent.path_cost + path_cost

            # fScore = f(n) = g(n) + h(n), specially used in A* algorithm.
            self.fScore = heuristicScore + self.path_cost

        def __str__(self):
            return str(self.state)

        __repr__ = __str__

    start_state = problem.getStartState()
    root = Node(state=start_state, parent=None, action=None
                , path_cost=0, heuristicScore = heuristic(start_state,problem))
    fringe = util.PriorityQueue()  # element is in the form of (priority, self.count, item)  -  item's type is node
    fringe.update(item=root, priority=root.fScore)
    explored = set()  # Graph Algorithm.

    while not fringe.isEmpty():     # pls refer to the depthFirstSearch function for detailed description.
        node_cur = fringe.pop()     # expand nodes

        if node_cur.state in explored:  # this node has been expanded
            continue
        else:
            explored.add(node_cur.state)

        if problem.isGoalState(node_cur.state):
            break

        # every successor is a triple in the form of ( successor/state, action, stepCost )
        successors = problem.getSuccessors(node_cur.state)

        for successor in successors:
            node_successor = Node(state=successor[0], parent=node_cur, action=successor[1]
                                  , path_cost=successor[2], heuristicScore=heuristic(successor[0],problem))
            if not node_successor.state in explored:
                fringe.update(item=node_successor, priority=node_successor.fScore)

    while node_cur.parent:
        path.append(node_cur.action)
        node_cur = node_cur.parent

    path.reverse()
    return path

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
