from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np

class HumanPlayer(Player):
    def __init__(self):
        self.side = None
        super().__init__()

    def move(self, board: Board):
        
        pass

    def final_result(self, result: int):
        pass

    def new_game(self, side: int):
        self.side = side