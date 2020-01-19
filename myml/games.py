import numpy as np
import matplotlib.pyplot as plt


class ParentError(Exception):
    pass


class Game(object):

    def __init__(self, name='Game'):
        if not isinstance(name, str):
            raise TypeError('"name" must be a string"')
        self.name = name
        self.board = None
        self.players = []

    def add_board(self, board):
        board._set_game(self)
        self.board = board

    def add_player(self, player):
        player._set_game(self)
        self.players.append(player)


class GameObject(object):

    def __init__(self, name=''):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self.name = name
        self.game = None

    def _set_game(self, game):
        if self.game:
            raise ParentError('This is already part of ' + self.game.name)
        else:
            self.game = game


class Player(GameObject):

    def _set_game(self, game):
        super()._set_game(game)
        self.idx = len(game.players)
        if not self.name:
            self.name = 'Player {0}'.format(self.idx)


class GameBoard(GameObject):

    def __init__(self, shape, nteams, base=None, teams=None):
        super().__init__()

        if not isinstance(shape, (list, tuple)):
            raise TypeError('"shape" must be a list/tuple')
        if not isinstance(nteams, int):
            raise TypeError('"nteams" must be an int')

        self.shape = shape
        self.board = np.zeros(shape, dtype=int)
        if isinstance(base, Tile):
            self.objects = [base]
        elif base is None:
            self.objects = [Tile()]
        else:
            raise TypeError('"base" must be a Tile type')

        if teams is None:
            self.teams = ['Team ' + str(i+1) for i in range(nteams)]
        elif not isinstance(names, (list, tuple)):
            raise TypeError('"teams" must be a list/tuple')
        elif nteams != len(teams) or nteams != len(set(teams)):
            raise ValueError('length of "teams" must be nteams and all items'
                             ' must be unique')
        else:
            self.teams = list(teams)
        self.pieces = {t:[] for t in self.teams}

    def _set_game(self, game):
        super()._set_game(game)
        self.name = game.name

    def _add_object(self, obj):
        obj._set_board(self)
        self.objects.append(obj)

    def check_pos(self, pos):
        if not isinstance(pos, (list, tuple)):
            raise TypeError('"pos" should be a list/tuple')
        elif len(pos) != len(self.shape):
            raise ValueError('"pos" must have length ' + str(len(self.shape)))
        elif all([0 <= p < s for p, s in zip(pos, self.shape)]):
            idx = self.board[pos]
            return self.objects[idx]
        else:
            return False

    def add_piece(self, piece, team, pos=None):
        self._add_object(piece)

        if isinstance(team, int):
            piece._set_team(self.teams[team])
        elif isinstance(team, str) and team in self.teams:
            piece._set_team(team)
        else:
            raise TypeError('"team" must be the integer representing the team'
                            ' or the name of the team')

        if not pos is None:
            piece._set_pos(pos)
            self.board[pos] = piece.idx


class GameAtom(object):

    def __init__(self, name='Piece'):
        if not isinstance(name, str):
            raise TypeError('"name" must be a string"')
        self.name = name
        self.board = None
        self.idx = None
        self.team = None
        self.pos = None

    def _set_board(self, board):
        if self.board:
            raise ParentError('This is already part of ' + self.board.name)
        else:
            self.board = board
            self.idx = len(board.objects)

    def _set_team(self, team):
        if self.team:
            raise ParentError('This is already part of ' + self.team)
        else:
            self.team = team

    def _set_pos(self, pos):
        check = self.board.check_pos(pos)
        if check and check.set_on(self):
            self.pos = pos
        elif not check:
            raise ValueError('This position is outwith the board')
        else:
            raise ValueError('This position is already in use')

    def land_on(self, piece):
        return True

    def set_on(self, piece):
        return True


class Tile(GameAtom):
    pass


class Piece(GameAtom):

    def land_on(self, piece):
        if piece.player == self.player:
            return False
        else:
            return True

    def set_on(self, piece):
        return False

    def move_valid(self, target):
        check = self.board.check_pos(target)
        if check:
            return check.land_on(self)
        else:
            return False
