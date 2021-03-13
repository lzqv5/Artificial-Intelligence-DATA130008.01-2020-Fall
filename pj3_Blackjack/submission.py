import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        # Counter example is just the same as the Problem 1
        return '0'
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return ["-1","1"]
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        def IsEnd(state):
            return state == "-2" or state == "2"
        if IsEnd(state):
            return []       # (newState, prob, reward) tuples
        if action == "-1":
            leftProb = 0.8
        else:
            leftProb = 0.7
        rightProb = 1 - leftProb
        if state == "-1":
            return [("-2",leftProb,20),("0",rightProb,-5)]
        elif state == "1":
            return [("0",leftProb,-5),("2",rightProb,100)]
        else:
            return [("-1",leftProb,-5),("1",rightProb,-5)]

        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues        # eg. [1,2,3] 表示一张牌有3类值 1,2,3   是一个列表
        self.multiplicity = multiplicity    # eg. 4 表示初始状态下,每张牌的个数      是一个正整数
        self.threshold = threshold          # eg. 10 表示手里的牌的值的总和不超过10   是个正整数
        self.peekCost = peekCost            # 单次偷看牌的成本.

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        totalCardValueInHand,nextCardIndexIfPeeked,deckCardCounts = state
        if deckCardCounts is None:  # 先判断是否为终止状态
            return []

        if action == "Quit": # return (newState, prob, reward)
            return [((totalCardValueInHand,nextCardIndexIfPeeked,None),1,totalCardValueInHand)]

        import numpy as np
        deckProb = list(np.array(deckCardCounts) / sum(deckCardCounts))
        def goBust(totalCardValueInHand,threshold):
            return totalCardValueInHand > threshold

        if action == "Peek" :
            if nextCardIndexIfPeeked != None: # can't peek twice in a row
                return []
            newStates = []
            for i in range(len(self.cardValues)):
                if deckProb[i]:  # 偷看到该卡牌的概率不为0
                    newState = (totalCardValueInHand, i, deckCardCounts)
                    newStates.append((newState, deckProb[i], -self.peekCost))
            return newStates

        # action == "Take"
        deckCardCounts = list(deckCardCounts)
        if nextCardIndexIfPeeked != None:   # 前一轮有peek过
            deckCardCounts[nextCardIndexIfPeeked] -= 1
            totalCardValueInHand += self.cardValues[nextCardIndexIfPeeked]
            if goBust(totalCardValueInHand,self.threshold): # bust
                newState = (totalCardValueInHand,None,None)
                return [(newState, 1, 0)]
            else: # safe
                if sum(deckCardCounts): # 牌堆非空
                    newState = (totalCardValueInHand,None,tuple(deckCardCounts))
                    return [(newState, 1, 0)]
                else:   # 牌堆已空
                    newState = (totalCardValueInHand, None, None)
                    return [(newState, 1, totalCardValueInHand)]

        else: # No peeking before taking
            newStates = []
            for i in range( len(self.cardValues) ):
                if deckProb[i]: # 表明该面值的卡还存在于牌堆中:
                    newTotalCardValueInHand = totalCardValueInHand + self.cardValues[i]
                    if goBust(newTotalCardValueInHand, self.threshold): # bust
                        newState = (newTotalCardValueInHand,None,None)
                        newStates.append((newState, deckProb[i], 0))
                    else:   # safe
                        deckCardCountsCopy = deckCardCounts[:]
                        deckCardCountsCopy[i] -= 1
                        if sum(deckCardCountsCopy): # deck is not empty
                            newState = (newTotalCardValueInHand, None, tuple(deckCardCountsCopy))
                            newStates.append((newState, deckProb[i], 0))
                        else:   # deck is empty
                            newState = (newTotalCardValueInHand, None, None)
                            newStates.append((newState, deckProb[i], newTotalCardValueInHand))
            return newStates
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # raise Exception("Not implemented yet")
    return BlackjackMDP(cardValues=[2,3,4,5,19], multiplicity=2,       # 面值大的牌当作爆牌，面值小的卡牌数量之间尽量地进行组合.
                                  threshold=20, peekCost=1)            # 使得小牌之间尽可能地组合而不会爆牌
    # return BlackjackMDP(cardValues=[4, 12, 17], multiplicity=20,  # 面值大的牌当作爆牌，面值小的卡牌数量之间尽量地进行组合.
    #                               threshold=20, peekCost=1)            # 使得小牌之间尽可能地组合而不会爆牌
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions                      # 每个状态可采取的后续动作
        self.discount = discount                    # 衰减因子
        self.featureExtractor = featureExtractor    # 特征提取器 - 将一个状态映射为一个特征向量
        self.explorationProb = explorationProb      # 策略提升时，选择探索动作的概率
        self.weights = defaultdict(float)           # 参数 - w
        # 这里的defaultdict的作用是,如果传入的key存在，则返回相对应的值.
        # 如果传入的参数不存在，则返回默认值 0.0
        # 那么显然,当特征提取器为identityFeatureExtractor时,对于那些
        # 在simulate时,没有被学习到的(s,a),RL对它们的预测值恒为0.
        self.numIters = 0                           # 迭代次数

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):      # 输入一个状态s和动作a，返回其Q(s,a)的近似值
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state. -- 相当于学习到的策略
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):         # 策略提升步骤， 将 s 映射为 a   (在QLearning中,被实现为 ε-greedy)
        self.numIters += 1
        if random.random() < self.explorationProb:      # exploration
            return random.choice(self.actions(state))
        else:                                           # exploitation
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):      # 获取学习率 α
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters. - 相当于传入一个转移过程-(s,a,r,s'),要求通过这个转移过程对我们的模型进行更新/学习
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        if newState is not None:    # 表示 state 不是终止状态.
            maxQvalue = max((self.getQ(newState, action), action) for action in self.actions(newState))[0]
            coefficient = self.getStepSize()*(reward + self.discount*maxQvalue - self.getQ(state,action))
            for f, v in self.featureExtractor(state, action):   # f - 索引, v - 特征值
                self.weights[f] += coefficient*v

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE

    # for the reproductivity
    random.seed(123)
    # initialization
    mdp.computeStates()     # to get the whole State Space of MDP
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, 0.2)
    # Value Iteration part
    algorithm = ValueIteration()
    algorithm.solve(mdp, .001)    # algorithm now contains the Value and Policy of the mdp.
    # Q-Learning part
    util.simulate(mdp, rl, 30000) # Model-Free, Simulate 30000 times. After this Q-learning has been learned.
    rl.explorationProb = 0        # set ε to 0 and then .getAction(state) works as a policy Π.
    Qpi = {}
    comparison = []
    for state in mdp.states:      # get the Q-learning policy and comparison results.
        Qpi[state]=rl.getAction(state)
        comparison.append(int(algorithm.pi[state]==Qpi[state]))
    if featureExtractor == identityFeatureExtractor:
        if mdp.multiplicity==2:
            print("The match rate of using identityFeatureExtractor for smallMDP: %.4f" % (sum(comparison) / len(comparison)))
            print("Number of different actions: %d  Number of total actions: %d" % ( len(comparison)-sum(comparison),len(comparison)))
        else:
            print("The match rate of using identityFeatureExtractor for largeMDP: %.4f" % (sum(comparison) / len(comparison)))
            print("Number of different actions: %d  Number of total actions: %d" % (len(comparison) - sum(comparison), len(comparison)))
    else:
        print("The match rate of using blackjackFeatureExtractor for largeMDP: %.4f" % (sum(comparison) / len(comparison)))
        print("Number of different actions: %d  Number of total actions: %d" % (len(comparison) - sum(comparison), len(comparison)))
    # 第二次的状态空间过大，在30000次的模拟过程中，有许多的状态被遇到，故对于它们的Q值的预测始终为0,
    # 那么Q-learning在它们上的策略是没有被学习到的，所以差别较大.
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    res = [((action, total),1)] # in the form of (featureKey,featureValue)
    if counts:
        counts = list(counts)
        for i in range(len(counts)):
            res.append(  ( (action, i, counts[i]), 1 )  )
            counts[i] = 1 if counts[i] >0 else 0
        res.append( ( (action,tuple(counts)),1 ) )
    return res
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    original_mdp.computeStates()  # to get the whole State Space of MDP
    modified_mdp.computeStates()
    algorithm = ValueIteration()
    algorithm.solve(original_mdp, .001)
    # algorithm.solve(modified_mdp, .001)

    frl = util.FixedRLAlgorithm(algorithm.pi)

    random.seed(123)
    totalRewards = util.simulate(mdp = modified_mdp, rl = frl, numTrials = 30)
    print( "*** Expected return for FixedRLAlgorithm (numTrials=30): %.4f \t***" % (sum(totalRewards)/len(totalRewards)) )
    totalRewards = util.simulate(mdp=modified_mdp, rl=frl, numTrials=30000)
    print(
        "*** Expected return for FixedRLAlgorithm (numTrials=30000): %.4f \t***" % (sum(totalRewards) / len(totalRewards)))
    random.seed(123)
    rlQ = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor)
    totalRewards = util.simulate(mdp = modified_mdp, rl = rlQ, numTrials = 30)  # Model-Free, Simulate 30000 times. After this Q-learning has been learned.
    print( "*** Expected return for QLearningRLAlgorithm (numTrials=30): %.4f \t ***" % (sum(totalRewards)/len(totalRewards)) )
    totalRewards = util.simulate(mdp=modified_mdp, rl=rlQ, numTrials=29970)
    print("*** Expected return for QLearningRLAlgorithm (numTrials=30000): %.4f \t ***" % (sum(totalRewards) / len(totalRewards)))
    # END_YOUR_CODE

