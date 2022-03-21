from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np
from copy import deepcopy


class BenchmarkPlayer(Player):
    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, board: Board):
        # get moves
        possible_moves = board.possible_actions()

        # shuffle moves
        np.random.shuffle(possible_moves)

        # find if one is winning
        for i in possible_moves:
            temp_board = deepcopy(board)
            move = (i[0], i[1])
            if temp_board.is_possible(move):
                temp_board.take_turn(move)
                over, winner = temp_board.is_over()
                if over:
                    board.take_turn(move)
                    over, winner = board.is_over()
                    return winner, board, over

        # otherwise return random move (first index)
        board.take_turn((possible_moves[0][0], possible_moves[0][1]))
        over, winner = board.is_over()
        return winner, board, over

    def final_result(self, result: int):
        pass

    def new_game(self, side: int):
        self.side = side
