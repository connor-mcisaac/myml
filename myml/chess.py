import numpy as np
from .games import *


class HomeTile(TeamTile):

    def _land_on(self, piece):
        if isinstance(piece, Pawn):
            """ Need to add calls to the player before implementing this """
            pass
        else:
            pass


class ChessBoard(GameBoard):

    def __init__(self):
        super().__init__((8, 8), 2, teams=['White', 'Black'])
        white = [[0, i] for i in range(8)]
        black = [[7, i] for i in range(8)]
        self.add_tiles(HomeTile(), white, 'White')
        self.add_tiles(HomeTile(), black, 'Black')
