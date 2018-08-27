import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        # the expected reward for taking action a from state s
        # it is python set, mutable containers of items of arbitrary types, with no duplicate
        # with s,a as parameter
        # e.g; self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        # e.g: self.Nsa[(s,a)] += 1
        self.Nsa = {}       # stores #times edge s,a was visited
        # e.g: self.Ns[s] += 1
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    # return will be some probability vector 
    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        # play many times, always start from the same state
        # so that we explore hopefully most valueble route to win
        for i in range(self.args.numMCTSSims):
            # search will play until leaf node was found
            self.search(canonicalBoard)
 
        s = self.game.stringRepresentation(canonicalBoard)
        # if action was not in the N(s,a), that means we never choosed that action
        # thus not useable in the future
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        # temperate 0 means we always choose the best route
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        # we return probabaly so coach/replay pit can use this to determine
        # which action to choose
        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    # The funtion returns either a new leaf node is found
    # or a terminated node is found.
    # The end of game was defined by game, maybe not real native end of the game
    # we then back propogate the Qsa(s,q), Nsa(s,a) etc
    #
    # Thus getActionProb() loop will start again from very beginning 
    # node to do the simulation again ( sort of BFS ) based on the new data
    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # s is hashRepresentation from current board state
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s]!=0:
            # terminal node, negative for the next player, as we play games 
            return -self.Es[s]

        # new node
        if s not in self.Ps:
            # what is a leaf node? a new node? 
            # remember we recursively call ourself
            # NN is take native state instead of s as input
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            # valid move from current state
            valids = self.game.getValidMoves(canonicalBoard, 1)
            #  a binary vector of length self.getActionSize(), 1 for
            # moves that are valid from the current board and player,
            # 0 for invalid moves
            self.Ps[s] = self.Ps[s]*valids      # masking invalid movesh

            sum_Ps_s = np.sum(self.Ps[s])
            # why? since we did mask?
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        # among valid move
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # got best estimated action
        a = best_act
        # what is next_s?
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        # why need this? is not next_s a state from current player or next_player?
        # always from certain point of view from certain player
        # the canonical form can be chosen to be from the pov 
        # returns canonical form of board. The canonical form 
        # should be independent of player. For e.g. in chess,
        # of white. When the player is white, we can return   
        # board as is. When the player is black, we can invert
        # the colors and return the board?
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # return reward from next player's view 
        v = self.search(next_s)

        # at the end of search, we update
        if (s,a) in self.Qsa:
            # this is an avg reward
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
