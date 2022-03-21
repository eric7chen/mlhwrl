from abc import ABC, abstractmethod
from typing import Tuple

from game.tic_tac_toe import Board


class Player(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, board: Board) -> Tuple[int, Board, bool]:
        pass

    @abstractmethod
    def final_result(self, result: int):
        pass

    @abstractmethod
    def new_game(self, side: int):
        pass
