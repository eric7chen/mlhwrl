import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from qlearn import QAgent

from game.tic_tac_toe import Board
from game.tic_tac_toe import TicTacToe
from rand_player import RandomPlayer
from game.Player import Player
from tdlearn import TDAgent
from tdlearn import learn as tdlearn

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
        if (over):
            player1.final_result(winner)
            player2.final_result(winner)
            return winner
        if (player1Turn):
            (winner, _, over) = player1.move(board)
        else:
            (winner, _, over) = player2.move(board)
        player1Turn = not player1Turn
        


def battle_q(player1: Player, player2: Player, num_games: int = 10000, silent: bool = False):
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
    print(f"after {num_games} game we have draws: {draw_count}, player1 wins: {wins_1}, player2 wins: {wins_2}")
    return (wins_1, wins_2, draw_count)

def eval_qplayers(p1 : Player, p2 : Player, games_per_battle = 100, num_battles = 100, silent: bool = False):
    p1_wins = []
    p2_wins = []
    draws = []
    count = []

    for i in range(num_battles):
        p1win, p2win, draw = battle_q(p1, p2, games_per_battle, silent)
        p1_wins.append(p1win*100.0/games_per_battle)
        p2_wins.append(p2win*100.0/games_per_battle)
        draws.append(draw*100.0/games_per_battle)
        count.append(i*games_per_battle)
      
    plt.figure()
    plt.plot(count, draws, label='Draws')
    plt.plot(count, p1_wins, label='Player 1 wins')
    plt.plot(count, p2_wins, label='Player 2 wins')
    plt.legend(shadow=True, fancybox=True, framealpha = 0.7)
    plt.ylabel('Outcome %')
    plt.xlabel('Training Iterations')
    plt.savefig("myplot3.png")

def battle_td(player1, player2, num_games):
    env = TicTacToe()
    player1.set_env(env)
    player2.set_env(env)
    first = True
    oneWins = 0
    twoWins = 0
    draws = 0
    env.reset()
    for i in range(num_games):
        while True:
            if first:
                action, greedy = player1.action()
                previous_state = env.board.hash()
                reward, _, over = env.step(action)
                current_state = env.board.hash()
            else:
                action, greedy = player2.action()
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
    return oneWins * 100.0 / num_games, twoWins * 100.0 / num_games, draws * 100.0 / num_games

def eval_tdplayers(games_per_battle = 100, num_battles = 200):
    ones = []
    twos = []
    draws = []
    count = []
    for i in range(0, 100, 5):
        one, two = tdlearn(i)
        one.reset_explore()
        two.reset_explore()
        a, b, c = battle_td(one, two, 1000)
        ones.append(a)
        twos.append(b)
        draws.append(c)
        count.append(i)

    print(ones, twos, draws, count)
    plt.plot(count, ones, color="red", label="Player One Win")
    plt.plot(count, twos, color="green", label="Player Two Win")
    plt.plot(count, draws, color="blue", label="Draw")
    plt.xlabel("Training Iterations")
    plt.ylabel("Outcome %")
    plt.legend()
    plt.savefig("myplottd.png")


#train q agent   
# player1 = QAgent()
# player2 = QAgent()
# eval_qplayers(player1, player2, 40, 200)

# train tdagent

eval_tdplayers(20, 100)

