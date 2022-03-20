#
# Copyright 2018 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from game.tic_tac_toe import Board
from game.Player import Player
import numpy as np


class RandomPlayer(Player):
    """
    This player can play a game of Tic Tac Toe by randomly choosing a free spot on the board.
    It does not learn or get better.
    """

    def __init__(self):
        """
        Getting ready for playing tic tac toe.
        """
        self.side = None
        super().__init__()

    def move(self, board: Board):
        """
        Making a random move
        :param board: The board to make a move on
        :return: The result of the move
        """
        index = np.random.randint(len(board.possible_actions()))
        for i in range(9):
            if board.is_possible((i // 3, i % 3)):
                if index == 0:
                    action = (i // 3, i % 3)
                    board.take_turn(action)
                    over, winner = board.is_over()
                    return winner, board, over
                else:
                    index -= 1

    def final_result(self, result: int):
        """
        Does nothing.
        :param result: The result of the game that just finished
        :return:
        """
        pass

    def new_game(self, side: int):
        """
        Setting the side for the game to come. Noting else to do.
        :param side: The side this player will be playing
        """
        self.side = side
