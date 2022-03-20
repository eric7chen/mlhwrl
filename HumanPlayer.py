from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np

class HumanPlayer(Player):
    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, board: Board):
        print(board.cells)
        move = int(input("Please enter a move:\n"))
        action = (move // 3, move % 3)
        while (not board.is_possible(action)):
            move = int(input("input again:\n"))
        action = (move // 3, move % 3)
        board.take_turn(action)
        over, winner = board.is_over()
        return winner, board, over

    def final_result(self, result: int):
        pass

    def new_game(self, side: int):
        self.side = side