from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np


class BenchmarkPlayer(Player):
    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, board: Board):
        index = np.random.randint(len(board.possible_actions()))
        possible_moves = board.possible_actions()
        np.random.shuffle(possible_moves)
        for i in possible_moves:
            temp_board = board
            if temp_board.is_possible(i):
                temp_board.take_turn(i)
                over, winner = board.is_over()
                if over:
                    board.take_turn(i)
                    return winner, temp_board, over

        return possible_moves[0]

    def final_result(self, result: int):
        pass

    def new_game(self, side: int):
        self.side = side
