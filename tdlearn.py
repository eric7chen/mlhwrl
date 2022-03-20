import argparse
import copy
from hashlib import new
import numpy
import random
from game.tic_tac_toe import Board
from game.tic_tac_toe import TicTacToe
from game.Player import Player

# rewards for win, draw, and loss
RES_WIN = 1.0
RES_DRAW = 0.5
RES_LOSS = 0.0


class TDAgent(Player):
    def __init__(self):
        self.side = None
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.vtable = numpy.zeros(3 ** (3 ** 2 + 1))
        # history for applying reward at the end of an episode
        self.history = []
    
    def get_move(self, board: Board) -> int:
        possible = board.possible_actions()
        # take random move epsilon % of the time
        if numpy.random.binomial(1, self.epsilon):
            move = tuple(random.choice(possible))
            return (3*move[0] + move[1])
        copies = [copy.deepcopy(board) for i in possible]
        for action, boardcopy in zip(possible, copies):
            boardcopy.take_turn(tuple(action))
        hashes = [boardcopy.hash() for boardcopy in copies]
        # move to state with best value
        move = possible[numpy.argmax(self.vtable[hashes])]
        return (3*move[0] + move[1])
    
    def move(self, board: Board):
        over, _ = board.is_over()
        move = self.get_move(board)
        action = (move // 3, move % 3)
        assert (board.is_possible(action))
        assert (not over)

        self.history.append(board.hash())
        board.take_turn(action)
        over, winner = board.is_over()
        return (winner, board, over)

    def reset_explore(self):
        self.exploration_rate = 0

    def set_env(self, environment):
        self.environment = environment
    
    def final_result(self, result: int):
        if (result == -1 and self.side == -1) or (result == 1 and self.side == 1):
            final_value = RES_WIN
        elif (result == -1 and self.side == 1) or (result == 1 and self.side == -1):
            final_value = RES_LOSS
        elif (result == 0):
            final_value = RES_DRAW
        else:
            raise ValueError("Unexpected game result")

        self.history.reverse()
        next_val = -1.0

        for h in self.history:
            # on first pass
            if next_val < 0:
                self.vtable[h] = final_value
            else:
                self.vtable[h] = self.vtable[h] + self.alpha * (final_value + self.gamma * next_state - self.vtable[h])
            next_state = self.vtable[h]
        


    def new_game(self, side: int):
        self.side = side
        self.history = []
