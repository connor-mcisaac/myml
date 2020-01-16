import numpy as np
import matplotlib.pyplot as plt


class EmptyTile(object):
    def land_on(self, other):
        return True


class GameBoard(object):

    def __init__(self, shape, name=None):
        if not isinstance(shape, (list, tuple)):
            raise TypeError('shape should be a list or tuple')
        self.shape = shape
        self.board = np.ones(shape, dtype=int) * -1
        self.npieces = 0
        self.pieces = {}
        self.nplayers = 0
        self.players = {}
        self.name = name if name else 'Game'
        self.empty = EmptyTile()

    def add_player(self, player):
        self.players[self.nplayers] = player
        self.nplayers += 1
        return self.nplayers - 1

    def check_pos(self, pos):
        if not isinstance(pos, (list, tuple)):
            raise TypeError('pos should be a list or tuple')
        if all([0 <= p < s for p, s in zip(pos, self.shape)]):
            return self.pieces.get(self.board[pos], self.empty)
        else:
            return False

    def add_piece(self, piece, pos):
        check = self.check_pos(pos)
        if not check:
            raise ValueError('This is not a valid position')
        elif isinstance(check, EmptyTile):
            self.board[pos] = self.npieces
            self.pieces[self.npieces] = piece
            self.npieces += 1
            return self.npieces - 1
        else:
            raise ValueError('This position is occupied')

    def move_piece(self, pos0, pos1):
        piece = self.check_pos(pos0)
        if isinstance(piece, EmptyTile):
            raise ValueError('pos0 does not contain a piece')
        elif piece.move_valid(pos1):
            self.board[pos0] = -1
            self.board[pos1] = piece.idx
            piece.pos = pos1
        else:
            raise ValueError('Cannot move piece to pos1')



class GamePlayer(object):

    def __init__(self, board, name):
        self.idx = board.add_player(self)
        self.board = board
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self.name = name


class GamePiece(object):

    def __init__(self, board, pos, player, name=None):
        self.idx = board.add_piece(self, pos)
        self.board = board
        self.pos = pos
        self.player = player
        self.name = name if name else 'Piece'

    def land_on(self, other):
        if other.player is self.player:
            return False
        else:
            return True

    def move_valid(self, target):
        check = self.board.check_pos(target)
        if check:
            return check.land_on(self)
        else:
            return False
