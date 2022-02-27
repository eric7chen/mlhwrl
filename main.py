import argparse
import copy
from hashlib import new
import numpy
import typing
import tqdm
import random


class Board(object):
    def __init__(self, cells: numpy.array = None) -> None:
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


def get_all_states() -> typing.Tuple[typing.Set, typing.Set]:
    boards = [Board()]
    states = set()
    terminal_states = set()
    epoch = 0
    while boards:
        print(f"Epoch: {epoch}")
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
    def __init__(self):
        self.board: Board = Board()

    def step(self, action: typing.Tuple[int, int]) -> typing.Tuple[int, Board, bool]:
        over, _ = self.board.is_over()
        # print(self.board.is_possible(action))
        assert self.board.is_possible(action)
        assert not over
        self.board.take_turn(action)
        over, winner = self.board.is_over()
        return winner, self.board, over

    def __repr__(self):
        return self.board.__repr__()

    def reset(self):
        self.__init__()


class Agent(object):
    def __init__(self, environment):
        self.environment = environment
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        self.values = numpy.zeros(3 ** (self.environment.board.size ** 2 + 1))

    def record(self, initial, resulting, reward, over):
        if over:
            self.values[resulting] = reward
        # V(s) = V(s) + alpha * [V(s') - V(s)]
        self.values[initial] = self.values[initial] + self.learning_rate * (
            self.values[resulting] - self.values[initial]
        )

    def action(self):
        possible = self.environment.board.possible_actions()
        if numpy.random.binomial(1, self.exploration_rate):
            return tuple(random.choice(possible)), False
        copies = [copy.deepcopy(self.environment.board) for i in possible]
        for action, board in zip(possible, copies):
            board.take_turn(tuple(action))
        hashes = [board.hash() for board in copies]
        return tuple(possible[numpy.argmax(self.values[hashes])]), True

    def reset_explore(self):
        self.exploration_rate = 0


def learn():
    env = TicTacToe()
    playerOne = Agent(env)
    playerTwo = Agent(env)
    iters = input("iterations?")
    episodes = range(int(iters))
    for i in episodes:
        playerOne_turn = True
        while True:
            if playerOne_turn:
                action, greedy = playerOne.action()
            else:
                action, greedy = playerTwo.action()
            playerOne_turn = not playerOne_turn
            prev_state = env.board.hash()
            reward, _, over = env.step(action)
            new_state = env.board.hash()
            if greedy:
                playerOne.record(prev_state, new_state, reward, over)
                playerTwo.record(prev_state, new_state, -reward, over)
            if over:
                env.reset()
                break
    return playerOne, playerTwo


def main():
    learn()


if __name__ == "__main__":
    main()
