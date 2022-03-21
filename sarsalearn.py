from typing import Tuple, final
from game.Player import Player
from game.tic_tac_toe import Board

import numpy as np

RES_WIN = 1.0
RES_DRAW = 0.5
RES_LOSS = 0.0

class SarsaAgent(Player):
    def __init__(self,
                 alpha: float =  0.9,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 q_init: float = 0.6) -> None:
        self.side = None
        self.training: bool = True

        # create qtable for each possible board state
        self.qtable = np.full((3 ** (3 ** 2 + 1), 9), q_init)

        self.history = []
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.q_init = q_init
        super().__init__()


    def get_move(self, board: Board) -> int:
        possible = board.possible_actions()
        # take random move epsilon % of the time
        if np.random.binomial(1, self.epsilon) and self.training:
            np.random.shuffle(possible)
            move = tuple(possible[0])
            return (3*move[0] + move[1])
        
        #otherwise, take best action from qtable
        board_hash = board.hash()
        qvals = self.qtable[board_hash]
        while True:
            move = np.argmax(qvals)
            if (board.is_possible((move // 3, move % 3))):
                return move
            else:
                qvals[move] = -1.0

    def move(self, board: Board):
        over, _ = board.is_over()
        move = self.get_move(board)
        action = (move // 3, move % 3)
        assert (board.is_possible(action))
        assert (not over)

        self.history.append((board.hash(), move))
        board.take_turn(action)
        over, winner = board.is_over()
        return (winner, board, over)

    def final_result(self, result: int):
        # if not training, dont update the q-table
        if self.training == False:
            return
        
        if (result == -1 and self.side == -1) or (result == 1 and self.side == 1):
            final_value = RES_WIN
        elif (result == -1 and self.side == 1) or (result == 1 and self.side == -1):
            final_value = RES_LOSS
        elif (result == 0):
            final_value = RES_DRAW

        self.history.reverse()
        firstTime = True

        for h in self.history:
            qvals = self.qtable[h[0]]
            if firstTime:  # First time through the loop
                # set the reward of the final move to result reward
                qvals[h[1]] = final_value
                firstTime = False
            else:
                # Sarsa Update
                # Q(S,A) = Q(S,A) + \alpha[R + \gamma*Q(S',A') - Q(S,A)]
                qvals[h[1]] = qvals[h[1]] + self.alpha * (self.gamma * next_q - qvals[h[1]])
            # next_q is the current state's action value pair
            next_q = qvals[h[1]]

    def new_game(self, side):
        self.side = side
        self.history = []

    def set_training(self, training):
        self.training = training