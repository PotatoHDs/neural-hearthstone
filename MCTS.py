import math
import copy
import numpy as np
from Game import GameImp


class MCTSNode():
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.Ns = 0
        self.Q = 0
        self.policy = None
        self.value = None
        self.valids = self.state.getValidMoves()
        return

    def expand(self, action):
        next_s, next_player = self.state.getNextState(1, action, self.state)
        # next_s = self.state.getState(self.state)
        child_node = MCTSNode(
            next_s, parent=self, action=action)

        self.children.append(child_node)
        return child_node

    def selectChild(self, a):
        for child in self.children:
            if child.action == a:
                return child
        return None

    def backpropagate(self, result):
        self.Ns += 1.
        self.Q = (self.Ns*self.Q + result) / self.Ns
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        # self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        # self.Nsa = {}  # stores #times edge s,a was visited
        # self.Ns = {}  # stores #times board s was visited
        # self.Ps = {}  # stores initial policy (returned by neural net)
        #
        # self.Es = {}  # stores game.getGameEnded ended for board s
        # self.Vs = {}  # stores game.getValidMoves for board s

    def buildTree(self, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        root = MCTSNode(copy.deepcopy(self.game.game))
        for i in range(self.args.numMCTS):
            self.search(root)

        s = self.game.stringRepresentation(self.game.game)
        counts = [root.selectChild((a,b)).Ns if root.selectChild((a,b)) else 0 for a in range(21) for b in range(18)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, node):
        while True:
            cur_best = -float('inf')
            best_act = -1
            for a in range(21):
                for b in range(18):
                    if node.valids[a, b]:
                        child = node.selectChild((a, b))
                        if child:
                            u = child.Q + node.policy[a, b] * math.sqrt(node.Ns) / (
                                    1 + child.Ns)
                        else:
                            u = node.policy[a, b] * math.sqrt(node.Ns + 1e-8)  # Q = 0 ?

                        if u > cur_best:
                            cur_best = u
                            best_act = (a, b)

            act = best_act

            newnode = node.selectChild(act)
            if newnode:
                node = newnode
            else:
                child_node = node.expand(act)
                child_node.value = child_node.state.getGameEnded()
                if child_node.state.ended or child_node.state.turn > 180:
                    # terminal node
                    child_node.backPropagate(-child_node.value)
                else:
                    child_node.policy, child_node.value = self.nnet.predict(child_node.state)
                    child_node.policy = child_node.policy * child_node.valids
                    sum_Ps_s = np.sum(child_node.policy)
                    if sum_Ps_s > 0:
                        child_node.policy /= sum_Ps_s  # renormalize
                    else:
                        print("All valid moves were masked, do workaround.")
                        child_node.policy = child_node.policy + child_node.valids
                        child_node.policy /= np.sum(child_node.policy)

                    child_node.backPropagate(-child_node.value)
                break


        # if s not in self.Es:
        #     self.Es[s] = self.game.getGameEnded(state, 1)
        # if self.Es[s] != 0:
        #     # terminal node
        #     return -self.Es[s]
        #
        # if s not in self.Ps:
        #     # leaf node
        #     self.Ps[s], v = self.nnet.predict(state)
        #     valids = self.game.getValidMoves(state, 1)
        #     self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
        #     sum_Ps_s = np.sum(self.Ps[s])
        #     if sum_Ps_s > 0:
        #         self.Ps[s] /= sum_Ps_s  # renormalize
        #     else:
        #         print("All valid moves were masked, do workaround.")
        #         self.Ps[s] = self.Ps[s] + valids
        #         self.Ps[s] /= np.sum(self.Ps[s])
        #
        #     self.Vs[s] = valids
        #     self.Ns[s] = 0
        #     return -v
        #
        # valids = self.Vs[s]
        # cur_best = -float('inf')
        # best_act = -1
        #
        # # pick the action with the highest upper confidence bound
        # for a in range(21):
        #     for b in range(18):
        #         if valids[a, b]:
        #             if (s, (a, b)) in self.Qsa:
        #                 u = self.Qsa[(s, (a, b))] + self.Ps[s][a, b] * math.sqrt(self.Ns[s]) / (
        #                             1 + self.Nsa[(s, (a, b))])
        #             else:
        #                 u = self.args.cpuct * self.Ps[s][a, b] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
        #
        #             if u > cur_best:
        #                 cur_best = u
        #                 best_act = (a, b)
        #
        # a = best_act
        # next_s, next_player = self.game.getNextState(state, 1, a)
        # next_s = self.game.getCanonicalForm(next_s, next_player)
        #
        # v = self.search(next_s)
        #
        # if (s, a) in self.Qsa:
        #     self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        #     self.Nsa[(s, a)] += 1
        #
        # else:
        #     self.Qsa[(s, a)] = v
        #     self.Nsa[(s, a)] = 1
        #
        # self.Ns[s] += 1
        # return -v
