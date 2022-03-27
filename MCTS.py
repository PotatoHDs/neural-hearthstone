import math
import copy
import numpy as np
from Game import GameImp


class MCTSNode:
    def __init__(self, state, game, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.Ns = 0
        self.Q = 0
        self.policy = None
        self.value = None
        self.valids = game.get_valid_moves()
        return

    def expand(self, game, action):
        next_s, next_player = game.get_next_state(1, action)
        # next_s = self.state.getState(self.state)
        child_node = MCTSNode(
            next_s, game, parent=self, action=action)

        self.children.append(child_node)
        return child_node

    def select_child(self, a):
        for child in self.children:
            if child.action == a:
                return child
        return None

    def back_propagate(self, result):
        self.Ns += 1.
        self.Q = (self.Ns * self.Q + result) / self.Ns
        if self.parent:
            self.parent.back_propagate(result)


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

    def get_action_prob(self, temp=1):
        self.root = MCTSNode(self.game.get_state(), copy.deepcopy(self.game))
        self.root.policy, self.root.value = self.nnet.predict(self.root.state)
        self.root.policy.shape = (21, 18)

        for i in range(self.args.numMCTS):
            self.search(self.root, copy.deepcopy(self.game))

        # s = self.game.stringRepresentation(state)
        counts = [self.root.select_child((a, b)).Ns if self.root.select_child((a, b)) else 0 for a in range(21) for b in
                  range(18)]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, node, game):
        while True:
            cur_best = -float('inf')
            best_act = -1
            if node.policy is None:
                print('Policy is none!')
            for a in range(21):
                for b in range(18):
                    if node.valids[a, b]:
                        child = node.select_child((a, b))
                        if child:
                            u = child.Q + node.policy[a, b] * math.sqrt(node.Ns) / (
                                    1 + child.Ns)
                        else:
                            u = node.policy[a, b] * math.sqrt(node.Ns + 1e-8)  # Q = 0 ?

                        if u > cur_best:
                            cur_best = u
                            best_act = (a, b)

            act = best_act

            newnode = node.select_child(act)
            if newnode:
                node = newnode
            else:
                child_node = node.expand(game, act)
                child_node.value = game.get_game_ended()
                if game.game.ended or game.game.turn > 180:
                    # terminal node
                    print('Terminal node')
                    child_node.back_propagate(-child_node.value)
                else:
                    child_node.policy, child_node.value = self.nnet.predict(child_node.state)
                    child_node.policy.shape = (21, 18)
                    child_node.policy = child_node.policy * child_node.valids
                    sum_ps_s = np.sum(child_node.policy)
                    if sum_ps_s > 0:
                        child_node.policy /= sum_ps_s  # renormalize
                    else:
                        print("All valid moves were masked, do workaround.")
                        child_node.policy = child_node.policy + child_node.valids
                        child_node.policy /= np.sum(child_node.policy)

                    child_node.back_propagate(-child_node.value)
                break