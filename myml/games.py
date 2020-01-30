import numpy as np


class ParentError(Exception):
    pass


class PositionError(Exception):
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
            self.name = 'Player {0}'.format(self.idx + 1)


class GameBoard(GameObject):

    def __init__(self, shape, nteams, base=None, teams=None):
        super().__init__()

        if not isinstance(shape, (list, tuple)):
            raise TypeError('"shape" must be a list/tuple')
        if not isinstance(nteams, int):
            raise TypeError('"nteams" must be an int')

        self.objects = []
        self.shape = shape
        self.board = np.zeros(shape, dtype=int)
        if isinstance(base, Tile):
            self._add_object(base)
        elif base is None:
            self._add_object(Tile())
        else:
            raise TypeError('"base" must be a Tile type')

        if teams is None:
            self.teams = ['Team ' + str(i+1) for i in range(nteams)]
        elif not isinstance(teams, (list, tuple)):
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

    def _validate_pos(self, pos):
        if not isinstance(pos, (list, tuple, np.ndarray)):
            raise TypeError('"pos" should be a list/tuple/array')
        elif len(pos) != len(self.shape):
            raise ValueError('"pos" must have length ' + str(len(self.shape)))

    def check_pos(self, pos):
        self._validate_pos(pos)
        if all([0 <= p < s for p, s in zip(pos, self.shape)]):
            idx = self.board[pos]
            return self.objects[idx]
        else:
            return False

    def add_tile(self, tile, pos, team=None):
        if tile not in self.objects:
            self._add_object(tile)

        tile._set_pos(pos)
        self.board[pos] = tile.idx

        if isinstance(team, int):
            piece._set_team(self.teams[team])
        elif team in self.teams or team is None:
            piece._set_team(team)

    def add_tiles(self, tile, pos, team=None):
        if not isinstance(pos, (list, tuple np.ndarray)):
            raise TypeError('"pos" should be a list/tuple/array')

        pos = np.array(pos)
        if len(pos.shape) != 2 or pos.shape[-1] != len(self.shape):
            raise ValueError('"pos" should be a list/tuple of lists/tuples '
                             'with length = ndim, or an array of shape '
                             '(npositions, ndim)')

        for i in range(pos.shape[0]):
            self.add_tile(tile, pos[i, :], team=team)

    def add_piece(self, piece, pos, team):
        self._add_object(piece)

        piece._set_pos(pos)
        self.board[pos] = piece.idx

        if isinstance(team, int):
            piece._set_team(self.teams[team])
        elif isinstance(team, str) and team in self.teams:
            piece._set_team(team)
        else:
            raise TypeError('"team" must be the integer representing the team'
                            ' or the name of the team')

    def _move_piece(self, p0, p1):
        piece = self.check_pos(p0)
        if not isinstance(piece, Piece):
            raise PositionError('Only pieces can be moved')
        move = piece._move_valid(p1)
        if move:
            self.board[p0] = piece.on.idx
            self.board[p1] = piece.idx
            piece._update_pos(p1, move)
        else:
            raise PositionError('Piece cannot be moved to this position')


class BoardObject(object):

    def __init__(self, name='Atom'):
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
        if team is not None:
            raise ValueError('This object cannot be given a team')

    def _set_pos(self, pos):
        pass

    def _save_state(self):
        return {}

    def _reset_state(self, state):
        pass

    def _can_land_on(self, piece):
        return True

    def _can_set_on(self, piece):
        return True

    def _land_on(self, piece):
        pass

    def _set_on(self, piece):
        pass


class Tile(BoardObject):
    pass


class TeamTile(BoardObject):

    def _set_team(self, team):
        if self.team:
            raise ParentError('This is already part of ' + self.team)
        else:
            self.team = team


class Piece(BoardObject):

    def __init__(self, name='Piece'):
        super().__init__(name=name)
        self.on = False
        self.alive = True

    def _set_team(self, team):
        if self.team:
            raise ParentError('This is already part of ' + self.team)
        else:
            self.team = team
        self.board.pieces[self.team].append(self)

    def _set_pos(self, pos):
        if self.pos:
            raise PositionError("This piece's position has already been set")
        check = self.board.check_pos(pos)
        if check and check._can_set_on(self):
            self.pos = pos
            check.set_on(self)
        elif not check:
            raise ValueError('This position is outwith the board')
        else:
            raise ValueError('This position is already in use')
        self.on = check

    def _save_state(self):
        return {'on': self.on,
                'alive': self.alive}

    def _reset_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def _can_land_on(self, piece):
        if piece.team == self.team:
            return False
        else:
            return True

    def _can_set_on(self, piece):
        return False

    def _land_on(self, piece):
        self.alive = False

    def _move_valid(self, target):
        check = self.board.check_pos(target)
        if check:
            return check._can_land_on(self)
        else:
            return False

    def _update_pos(self, pos, in_pos):
        self.pos = pos
        if isinstance(in_pos, Tile):
            self.on = in_pos
        elif isinstance(in_pos, Piece):
            self.on = in_pos.on
        in_pos._land_on(self)
