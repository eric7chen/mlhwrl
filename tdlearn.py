import argparse
import copy
from hashlib import new
import numpy
import random
from game.tic_tac_toe import Board
from game.tic_tac_toe import TicTacToe
from game.Player import Player


class TDAgent(Player):
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

    def set_env(self, environment):
        self.environment = environment
    
    def move(self, env: Board):
        possible = env.possible_actions()
        copies = [copy.deepcopy(env) for i in possible]
        for action, board in zip(possible, copies):
            board.take_turn(tuple(action))
        hashes = [board.hash() for board in copies]

        env.take_turn(tuple(possible[numpy.argmax(self.values[hashes])]))
        over, winner = env.is_over()
        return (winner, env, over)
    
    def final_result(self, result: int):
        pass

    def new_game(self, side: int):
        pass


def learn(iters):
    env = TicTacToe()
    playerOne = TDAgent(env)
    playerTwo = TDAgent(env)
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


def play(one, two, games):
    env = TicTacToe()
    playerOne = one
    playerTwo = two
    playerOne.set_env(env)
    playerTwo.set_env(env)
    iters = games
    first = True
    oneWins = 0
    twoWins = 0
    draws = 0
    env.reset()
    for i in range(iters):
        while True:
            if first:
                action, greedy = playerOne.action()
                previous_state = env.board.hash()
                reward, _, over = env.step(action)
                current_state = env.board.hash()
            else:
                action, greedy = playerTwo.action()
                previous_state = env.board.hash()
                reward, _, over = env.step(action)
                current_state = env.board.hash()
            first = not first
            # print(env.board)
            if over:
                if (reward == 1 and first) or (reward == -1 and not first):
                    oneWins += 1
                elif (reward == -1 and first) or (reward == 1 and not first):
                    twoWins += 1
                else:
                    draws += 1
                # print(env.board)
                env.reset()
                break
    return oneWins * 100.0 / games, twoWins * 100.0 / games, draws * 100.0 / games

