import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from HumanPlayer import HumanPlayer
from qlearn import QAgent

from game.tic_tac_toe import Board
from game.tic_tac_toe import TicTacToe
from rand_player import RandomPlayer
from game.Player import Player
from sarsalearn import SarsaAgent
from tdlearn import TDAgent
from benchmark_player import BenchmarkPlayer

PLAYER1 = 1
PLAYER2 = -1


def play_game(board: Board, player1: Player, player2: Player) -> int:
    player1Turn = True
    player1.new_game(PLAYER1)
    player2.new_game(PLAYER2)
    board = Board()
    winner = 0
    over = False
    while True:
        if over:
            player1.final_result(winner)
            player2.final_result(winner)
            return winner
        if player1Turn:
            (winner, board, over) = player1.move(board)
        else:
            (winner, board, over) = player2.move(board)
        player1Turn = not player1Turn


def battle(
    player1: Player, player2: Player, num_games: int = 10000, silent: bool = False
):
    board = Board()
    draw_count = 0
    wins_1 = 0
    wins_2 = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2)
        if result == PLAYER1:
            wins_1 += 1
        elif result == PLAYER2:
            wins_2 += 1
        else:
            draw_count += 1
    print(
        f"after {num_games} game we have draws: {draw_count}, player1 wins: {wins_1}, player2 wins: {wins_2}"
    )
    return (wins_1, wins_2, draw_count)


def eval_players(
    output,
    p1: Player,
    p2: Player,
    games_per_battle=100,
    num_battles=100,
    silent: bool = False,
):
    p1_wins = []
    p2_wins = []
    draws = []
    count = []

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, silent)
        p1_wins.append(p1win * 100.0 / games_per_battle)
        p2_wins.append(p2win * 100.0 / games_per_battle)
        draws.append(draw * 100.0 / games_per_battle)
        count.append(i * games_per_battle)

    plt.figure()
    plt.plot(count, draws, label="Draws")
    plt.plot(count, p1_wins, label="Player 1 wins")
    plt.plot(count, p2_wins, label="Player 2 wins")
    plt.legend(shadow=True, fancybox=True, framealpha=0.7)
    plt.ylabel("Outcome %")
    plt.xlabel("Training Iterations")
    plt.savefig(output + ".png")


# train q agent
# player1 = QAgent()
# player2 = QAgent()
# eval_players("qvq", player1, player2, 40, 200)

# play q agent against benchmark
# player1.set_training(False)
# player2 = BenchmarkPlayer()
# eval_players("qvqanalysis", player1, player2, 40, 200)

# train q agent against benchmark
# player1 = QAgent()
# player2 = BenchmarkPlayer()

# play q agent against benchmark
# player1.set_training(False)
# eval_players("qvb", player1, player2, 40, 200)
# player2 = BenchmarkPlayer()
# eval_players("qvbanalysis", player1, player2, 40, 200)

# train tdagent
# player1 = TDAgent()
# player2 = TDAgent()
# eval_players('tdvtd',player1, player2, 10, 40)

# player1.set_training(False)
# player2 = HumanPlayer()
# eval_players('tdvh', player1, player2, 1, 3)

# play td agent aginst benchmark
# player1.set_training(False)
# player2 = BenchmarkPlayer()
# eval_players("tdvtd analysis", player1, player2, 40, 200)

# train sarsa agent
player1 = SarsaAgent()
player2 = SarsaAgent()
eval_players('svs', player1, player2, 40, 200)

player1.set_training(False)
player2 = BenchmarkPlayer()
eval_players('svb', player1, player2, 40, 200)