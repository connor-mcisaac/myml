import numpy as np
from games import *


class ChessBoard(GameBoard):

    def __init__(self):
        super().__init__((8, 8), 2, teams=['White', 'Black'])
