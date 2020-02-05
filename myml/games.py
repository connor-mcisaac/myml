import numpy as np


class ParentError(Exception):
    pass


class PositionError(Exception):
    pass


class TeamError(Exception):
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

        self.nteams = nteams
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
        self.turn = 0

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
        return tuple(pos)

    def _validate_team(self, team):
        if isinstance(team, int):
            return self.teams[team]
        elif isinstance(team, str) and team in self.teams:
            return team
        elif team is None:
            return team
        else:
            raise TypeError('"team" must be the integer representing the team'
                            ' or the name of the team')

    def _check_pos(self, pos):
        if all([0 <= p < s for p, s in zip(pos, self.shape)]):
            idx = self.board[pos]
            return self.objects[idx]
        else:
            return False

    def add_tile(self, tile, pos, team=None):
        pos = self._validate_pos(pos)
        team = self._validate_team(team)

        if tile not in self.objects:
            self._add_object(tile)

        tile._set_pos(pos)
        tile._set_team(team)

        self.board[pos] = tile.idx

    def add_tiles(self, tile, pos, team=None):
        if not isinstance(pos, (list, tuple, np.ndarray)):
            raise TypeError('"pos" should be a list/tuple/array')

        pos = np.array(pos)
        if len(pos.shape) != 2 or pos.shape[-1] != len(self.shape):
            raise ValueError('"pos" should be a list/tuple of lists/tuples '
                             'with length = ndim, or an array of shape '
                             '(npositions, ndim)')

        if isinstance(team, int):
            tile._set_team(self.teams[team])
        elif team in self.teams:
            tile._set_team(team)

        for i in range(pos.shape[0]):
            self.add_tile(tile, pos[i, :])

    def add_piece(self, piece, pos, team):
        pos = self._validate_pos(pos)
        team = self._validate_team(team)

        self._add_object(piece)
        try:
            piece._set_pos(pos)
        except PositionError:
            del self.objects[-1]
        piece._set_team(team)
        self.board[pos] = piece.idx

    def move_piece(self, p0, p1):
        p0 = self._validate_pos(p0)
        p1 = self._validate_pos(p1)
        team = self.teams[self.turn % self.nteams]
        piece = self._check_pos(p0)
        if not isinstance(piece, Piece):
            raise PositionError('Only pieces can be moved')
        elif team != piece.team:
            raise TeamError("This piece doesn't belong to " + team)
        move = piece._move_valid(p1)
        if move:
            on = piece.on.idx
            piece._update_pos(p1)
            self.board[p0] = on
            self.board[p1] = piece.idx
        else:
            raise PositionError('Piece cannot be moved to this position')
        self.turn += 1

    def _save_state(self):
        return {'turn': self.turn,
                'board': self.board.copy(),
                'objects': self.objects.copy()}

    def _save_full_state(self):
        state = self._save_state()
        state['states'] = [ob._save_state() for ob in self.objects]
        return state

    def _reset_state(self, state):
        for k, v in state.items():
            if k != 'states':
                setattr(self, k, v)
        if 'states' in state.keys():
            for ob, ob_state in zip(self.objects, state['states']):
                ob._reset_state(ob_state)

    def _check_board_state(self):
        return True

    def _try_move(self, p0, p1):
        state = self._save_full_state()
        try:
            self.move_piece(p0, p1)
        except TeamError:
            return False
        allowed = self._check_board_state()
        self._reset_state(state)
        return allowed

    def _list_piece_moves(self, piece):
        if isinstance(piece, int):
            piece = self.objects[piece]
        moves = []
        for m in piece._list_moves():
            if self._try_move(piece.pos, m):
                moves.append((piece.pos, m))
        return moves

    def _list_pos_moves(self, pos):
        pos = self._validate_pos(pos)
        check = self._check_pos(pos)
        return self._list_piece_moves(check)

    def _list_moves(self):
        team = self.teams[self.turn % self.nteams]
        moves = []
        for piece in self.pieces[team]:
            for m in piece._list_moves():
                if self._try_move(piece.pos, m):
                    moves.append((piece.pos, m))
        return moves


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
        check = self.board._check_pos(pos)
        if not check:
            raise PositionError('This position is outwith the board')

    def _save_state(self):
        return {}

    def _reset_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def _can_land_on(self, piece):
        return True

    def _can_set_on(self, piece):
        return True

    def _land_on(self, piece):
        pass

    def _set_on(self, piece):
        pass

    def _list_moves(self):
        return []


class Tile(BoardObject):
    pass


class TeamTile(Tile):

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
        check = self.board._check_pos(pos)
        if check and check._can_set_on(self):
            self.pos = tuple(pos)
            check._set_on(self)
        elif not check:
            raise PositionError('This position is outwith the board')
        else:
            raise PositionError('This position is already in use')
        self.on = check

    def _save_state(self):
        return {'pos': self.pos,
                'on': self.on,
                'alive': self.alive}

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
        check = self.board._check_pos(target)
        if check:
            return check._can_land_on(self)
        else:
            return False

    def _update_pos(self, pos):
        in_pos = self.board._check_pos(pos)
        if isinstance(in_pos, Tile):
            self.on = in_pos
        elif isinstance(in_pos, Piece):
            self.on = in_pos.on
        in_pos._land_on(self)
        self.pos = pos

    def _list_moves(self):
        moves = []
        for i in range(self.board.board.shape[0]):
            for j in range(self.board.board.shape[1]):
                if self._move_valid((i, j)):
                    moves.append((i, j))
        return moves
