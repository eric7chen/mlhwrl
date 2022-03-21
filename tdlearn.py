import argparse
import copy
from hashlib import new
from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np

# rewards for win, draw, and loss
RES_WIN = 1.0
RES_DRAW = 0.5
RES_LOSS = 0.0


class TDAgent(Player):
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        v_init: float = 0.05,
    ) -> None:
        self.side = None
        self.training: bool = True
        # create table for each possible board state
        self.vtable = np.empty([3 ** (3 ** 2 + 1)])
        self.vtable.fill(v_init)
        self.history = []
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.v_init = v_init
        super().__init__()

    def get_move(self, board: Board) -> int:
        possible = board.possible_actions()

        # take random move epsilon % of the time
        if np.random.binomial(1, self.epsilon) and self.training:
            np.random.shuffle(possible)
            move = tuple(possible[0])
            return 3 * move[0] + move[1]

        # otherwise take best move from possible choices
        copies = [copy.deepcopy(board) for i in possible]
        for action, boardcopy in zip(possible, copies):
            boardcopy.take_turn(tuple(action))
        hashes = [boardcopy.hash() for boardcopy in copies]
        move = tuple(possible[np.argmax(self.vtable[hashes])])
        return 3 * move[0] + move[1]

    def move(self, board: Board):
        over, _ = board.is_over()
        move = self.get_move(board)
        action = (move // 3, move % 3)
        assert board.is_possible(action)
        assert not over

        self.history.append(board.hash())
        board.take_turn(action)
        over, winner = board.is_over()
        return (winner, board, over)

    def final_result(self, result: int):
        if (result == -1 and self.side == -1) or (result == 1 and self.side == 1):
            final_value = RES_WIN
        elif (result == -1 and self.side == 1) or (result == 1 and self.side == -1):
            final_value = RES_LOSS
        elif result == 0:
            final_value = RES_DRAW

        self.history.reverse()
        firstTime = True

        for state in self.history:
            # on first pass
            if firstTime:
                self.vtable[state] = final_value
                firstTime = False
            else:
                # V(S) = V(S) + \alpha[R + \gamma*V(S') - V(S)]
                self.vtable[state] = self.vtable[state] + self.alpha * (
                    self.gamma * next_state - self.vtable[state]
                )
            # next_state is the value for the current state
            next_state = self.vtable[state]

    def new_game(self, side: int):
        self.side = side
        self.history = []

    def set_training(self, training):
        self.training = training
