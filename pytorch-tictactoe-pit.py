import Arena
from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.pytorch.NNet import NNetWrapper as NNet 
from tictactoe.TicTacToePlayers import * 


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TicTacToeGame() 

# all players
rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
#hp = HumanOthelloPlayer(g).play
hp = HumanTicTacToePlayer(g).play

# nnet players
n1 = NNet(g)
#n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
n1.load_checkpoint('./pretrained_models/tictactoe/keras/','best-25eps-25sim-10epch.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
