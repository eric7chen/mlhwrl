import argparse
import copy
import numpy
import typing
import tqdm


class Board(object):
    def print_board(self):
        print(self.cells)

    def __init__(self, cells: numpy.array = None) -> None:
        # Use classic 3x3 size.
        self.size: int = 3
        self.first_player_turn: bool = True
        self.cells: numpy.array
        if cells is not None:
            assert cells.shape == (self.size, self.size)
            self.cells = cells
        else:
            self.cells = numpy.zeros((self.size, self.size), dtype=numpy.int8)

    def take_turn(self, cell: typing.Tuple[int, int]):
        assert self.is_possible(cell)
        player_identifier = 1
        if not self.first_player_turn:
            player_identifier = -1
        self.cells[cell] = player_identifier
        # Switch current player after the turn.
        self.first_player_turn = not self.first_player_turn

    def is_possible(self, action: typing.Tuple[int, int]) -> bool:
        return self.cells[action] == 0

    def possible_actions(self) -> numpy.array:
        return numpy.array(
            [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
                if self.is_possible((i, j))
            ]
        )

    def is_over(self) -> typing.Tuple[bool, int]:
        # Check for all horizontal sequences of 3 consequent non-empty cells
        for i in range(self.size):
            OK = True
            player_id = self.cells[i][0]
            if player_id == 0:
                continue
            for j in range(self.size):
                if self.cells[i][j] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Vertical sequences
        for i in range(self.size):
            OK = True
            player_id = self.cells[0][i]
            if player_id == 0:
                continue
            for j in range(self.size):
                if self.cells[j][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Diagonal: left top to right bottom
        OK = True
        player_id = self.cells[0][0]
        if player_id != 0:
            for i in range(self.size):
                if self.cells[i][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # Diagonal: left bottom to right top
        OK = True
        player_id = self.cells[self.size - 1][0]
        if player_id != 0:
            for i in range(self.size):
                if self.cells[self.size - i - 1][i] != player_id:
                    OK = False
            if OK:
                return True, player_id
        # If there is an empty cell, the game is not over yet.
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] == 0:
                    return False, 0
        # Otherwise all cells are taken and no player has won: it's a draw!
        return True, 0

    def hash(self) -> int:
        result = 0
        for i in range(self.size):
            for j in range(self.size):
                result *= 3
                result += self.cells[i][j] % 3
        return result

    def __repr__(self) -> str:
        result = ""
        mapping = [" ", "X", "O"]
        for i in range(self.size):
            for j in range(self.size):
                result += " {} ".format(mapping[self.cells[i][j]])
                if j != self.size - 1:
                    result += "|"
                else:
                    result += "\n"
            if i != self.size - 1:
                result += ("-" * (2 + self.size * self.size)) + "\n"
        return result