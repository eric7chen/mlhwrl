'''
Author: Kirill Bobyrev (https://github.com/kirillbobyrev)

This module implements "An Extended Example: Tic Tac Toe" from `Reinforcement
Learning: An Introduction`_ book by Richard S. Sutton and Andrew G. Barto
(January 1, 2018 complete draft) described in Section 1.5. The implemented
Reinforcement Learning algorithm is TD(0) and it is trained via self-play
between two agents. The update rule is slightly modified given the environment
specifics to comply with the one introduced in the Chapter 1, but as shown
later is equivalent to the one used in generic settings.

Example:
    In order to run this script you would require a recent Python 3 interpreter
    (versions 3.6 and newer) and few PyPi packages (numpy, tqdm). To train a
    TD(0) agent and launch an interactive session to play against the AI simply
    run::

        $ python tic_tac_toe.py

    If you would like to take the first turn against the AI run::

        $ python tic_tac_toe.py --take_first_turn

    Learning the policy for the Reinforcement Learning action would take around
    a minute by default (20000 episodes), use --episodes to alter the number
    of training simulations::

        $ python tic_tac_toe.py --episodes 1000

.. _Reinforcement Learning: An Introduction:
   http://incompleteideas.net/book/the-book-2nd.html
'''

import argparse
import copy
import numpy
import typing
import tqdm


class Board(object):
    '''
    The classic 3 by 3 Tic Tac Toe board interface implementation, which is
    used as a part of Reinforcement Learning environment. It provides the
    necessary routine methods for accessing the internal states and allows
    safely modifying it while maintaining a valid state.

    Cell coordinates are zero-based indices: (x, y). Top left cell's
    coordinates are (0, 0), bottom right - (2, 2), i.e. the whole board looks
    like this:

     (0, 0) | (0, 1) | (0, 2)
    --------------------------
     (1, 0) | (1, 1) | (1, 2)
    --------------------------
     (2, 0) | (2, 1) | (2, 2)
    '''

    def __init__(self, cells: numpy.array = None) -> None:
        # Use classic 3x3 size.
        self.size: int = 3
        self.first_player_turn: bool = True
        self.cells: numpy.array
        if cells is not None:
            assert (cells.shape == (self.size, self.size))
            self.cells = cells
        else:
            self.cells = numpy.zeros((self.size, self.size), dtype=numpy.int8)

    def take_turn(self, cell: typing.Tuple[int, int]):
        '''
        Modifies current board given player's decision.

        Expects given cell to be empty, otherwise produces an exception.
        '''
        assert (self.is_possible(cell))
        player_identifier = 1
        if not self.first_player_turn:
            player_identifier = -1
        self.cells[cell] = player_identifier
        # Switch current player after the turn.
        self.first_player_turn = not self.first_player_turn

    def is_possible(self, action: typing.Tuple[int, int]) -> bool:
        '''
        Checks whether an action is valid on this board.

        Args:
            action: Coordinates of the action to check for validity.

        Returns:
            bool: True if it is possible to put 'X' or 'O' into the given cell,
                False otherwise.
        '''
        return self.cells[action] == 0

    def possible_actions(self) -> numpy.array:
        '''
        Outputs a all possible actions from current board state by choosing the
        ones not previously taken by either player.

        Returns:
            numpy.array: An array of possible actions.
        '''
        return numpy.array([(i, j)
                            for i in range(self.size)
                            for j in range(self.size)
                            if self.is_possible((i, j))])

    def is_over(self) -> typing.Tuple[bool, int]:
        '''
        Determines whether the game is over and hence no possible further
        action can be taken by either side.

        Returns:
            bool: True if the game is over, False otherwise.
            int: If the game is over, returns identifier of the winner (1 or
                -1 for the first and the second player respectively), 0
                otherwise.
        '''
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
        '''
        Bijectively maps board state to its unique identifier.

        Returns:
            int: Unique identifier of the current Board state.
        '''
        result = 0
        for i in range(self.size):
            for j in range(self.size):
                result *= 3
                result += self.cells[i][j] % 3
        return result

    def __repr__(self) -> str:
        '''
        Returns the Tic Tac Toe board in a human-readable representation using
        the following form (indices are replaced with 'X's, 'O's and
        whitespaces for empty cells):

         0 | 1 | 2
        -----------
         3 | 4 | 5
        -----------
         6 | 7 | 8
        '''
        result = ''
        mapping = [' ', 'X', 'O']
        for i in range(self.size):
            for j in range(self.size):
                result += ' {} '.format(mapping[self.cells[i][j]])
                if j != self.size - 1:
                    result += '|'
                else:
                    result += '\n'
            if i != self.size - 1:
                result += ('-' * (2 + self.size * self.size)) + '\n'
        return result


def get_all_states() -> typing.Tuple[typing.Set, typing.Set]:
    '''
    Devises all valid board states and computes hashes for each of them. Also
    extracts terminal states useful for the update rule simplification.

    Returns:
        set: A set of all possible boards' hashes.
        set: A set of hashes of all boards after a final turn, i.e. terminal
            boards.
    '''
    boards = [Board()]
    states = set()
    terminal_states = set()
    epoch = 0
    while boards:
        print(f'Epoch: {epoch}')
        epoch += 1
        next_generation = []
        for board in boards:
            board_hash = board.hash()
            if board_hash in states:
                continue
            states.add(board_hash)
            over, _ = board.is_over()
            if over:
                terminal_states.add(board_hash)
                continue
            for action in board.possible_actions():
                next_board = copy.deepcopy(board)
                next_board.take_turn(tuple(action))
                next_generation.append(next_board)
        boards = next_generation
    return states, terminal_states


class TicTacToe(object):
    '''
    TicTacToe is a Reinforcement Learning environment for this game, which
    reacts to players' moves, updates the internal state (Board) and samples
    reward.
    '''

    def __init__(self):
        self.board: Board = Board()

    def step(self,
             action: typing.Tuple[int, int]) -> typing.Tuple[int, Board, bool]:
        '''
        Updates the board given a valid action of the current player.

        Args:
            action: A valid action in a form of cell coordinates.

        Returns:
            int: Reward for the first player.
            Board: Resulting state.
            bool: True if the game is over, False otherwise.
        '''
        over, _ = self.board.is_over()
        assert (self.board.is_possible(action))
        assert (not over)
        self.board.take_turn(action)
        over, winner = self.board.is_over()
        return winner, self.board, over

    def __repr__(self):
        '''
        Returns current board state using a human-readable string
        representation.
        '''
        return self.board.__repr__()

    def reset(self):
        '''
        Empties the board and starts a new game.
        '''
        self.__init__()