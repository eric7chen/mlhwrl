from typing import Tuple, final
from game.Player import Player
from game.tic_tac_toe import Board

import numpy as np

RES_WIN = 1.0
RES_DRAW = 0.5
RES_LOSS = 0.0

class QAgent(Player):
    def __init__(self,
                 alpha: float =  0.9,
                 gamma: float = 0.9,
                 q_init: float = 0.6) -> None:
        self.side = None
        # create qtable for each possible board state
        self.qtable = []
        for i in range(3 ** (3 ** 2 + 1)):
            self.qtable.append(np.full(9, q_init))
        self.history = []
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.q_init = q_init
        super().__init__()


    def get_move(self, board: Board) -> int:
        board_hash = board.hash()
        qvals = self.qtable[board_hash]
        while True:
            move = np.argmax(qvals)
            if (board.is_possible((move // 3, move % 3))):
                return move
            else:
                qvals[move] = -1.0

    def move(self, board: Board):
        over, _ = board.is_over()
        move = self.get_move(board)
        action = (move // 3, move % 3)
        assert (board.is_possible(action))
        assert (not over)

        self.history.append((board.hash(), move))
        board.take_turn(action)
        over, winner = board.is_over()
        return (winner, board, over)

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
        next_max = -1.0

        for h in self.history:
            qvals = self.qtable[h[0]]
            if next_max < 0:  # First time through the loop
                qvals[h[1]] = final_value
            else:
                qvals[h[1]] = qvals[h[1]] * (
                    1.0 - self.alpha) + self.alpha * self.gamma * next_max
            next_max = max(qvals)

    def new_game(self, side):
        self.side = side
        self.history = []