import numpy as np


class GameBoard(GameObject):

    def __init__(self, shape, nteams, base=None, teams=None):
        super().__init__()

        if not isinstance(shape, (list, tuple)):
            raise TypeError('"shape" must be a list/tuple')
        if not isinstance(nteams, int):
            raise TypeError('"nteams" must be an int')

        self.objects = []
        self.shape = tuple(shape)
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
        self.state = None
        self.pos_states = {}

    def __repr__(self):
        letters = len(_number_to_letters(self.shape[1]))
        pieces = np.max([len(str(b)) for b in self.objects])
        letters = max(letters, pieces)
        numbers = len(str(self.shape[0]))
        rep = ' ' * numbers + '  '
        for i in range(self.shape[1]):
            rep += ' {0}  '.format(_number_to_letters(i).ljust(letters))
        line = ' ' * (numbers + 2)
        line += '_' * (len(rep) - len(line))
        rep += '\n' + line + '\n'
        for i in range(self.shape[0]):
            rep += '{0} |'.format(str(i).ljust(numbers))
            for j in range(self.shape[1]):
                piece = str(self.objects[self.board[i, j]])
                rep += ' {0} |'.format(piece.ljust(letters))
            rep += '\n' + line + '\n'
        return rep

    def _set_game(self, game):
        super()._set_game(game)
        self.name = game.name

    def _add_object(self, obj, pos=None, team=None):
        obj._set_board(self)
        obj._set_pos(pos)
        obj._set_team(team)
        obj.idx = len(self.objects)
        if pos:
            self.board[pos] = obj.idx
        self.objects.append(obj)

    def _validate_pos(self, pos):
        if not isinstance(pos, (list, tuple, np.ndarray)):
            raise TypeError('"pos" should be a list/tuple/array')
        elif len(pos) != len(self.shape):
            raise ValueError('"pos" must have length ' + str(len(self.shape)))
        return tuple(pos)

    def _within_board(self, pos):
        return all([0 <= p < s for p, s in zip(pos, self.shape)])

    def _get_object(self, pos):
        return self.objects[self.board[pos]]

    def _check_pos(self, pos):
        if self._within_board(pos):
            return self._get_object(pos)
        else:
            return False

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

    def add_tile(self, tile, pos, team=None):
        pos = self._validate_pos(pos)
        if not self._within_board(pos):
            raise PositionError("Can't add a tile outwith board")
        team = self._validate_team(team)

        if tile not in self.objects:
            self._add_object(tile, pos=pos, team=team)
        else:
            self.board[pos] = tile.idx

    def add_tiles(self, tile, pos, team=None):
        if not isinstance(pos, (list, tuple, np.ndarray)):
            raise TypeError('"pos" should be a list/tuple/array')

        pos = np.array(pos)
        if len(pos.shape) != 2 or pos.shape[-1] != len(self.shape):
            raise ValueError('"pos" should be a list/tuple of lists/tuples '
                             'with length = ndim, or an array of shape '
                             '(npositions, ndim)')

        for i in range(pos.shape[0]):
            self.add_tile(tile, pos[i, :], team=team)

    def add_piece(self, piece, pos, team):
        pos = self._validate_pos(pos)
        if not self._within_board(pos):
            raise PositionError("Can't add a piece outwith board")
        team = self._validate_team(team)

        self._add_object(piece, pos=pos, team=team)
        self.pieces[team].append(piece)

    def _move_piece(self, p0, p1):
        piece = self._get_object(p0)
        on = piece.on.idx
        piece._update_pos(p1)
        self.board[p0] = on
        self.board[p1] = piece.idx

    def _whose_turn(self):
        return self.teams[self.turn % self.nteams]

    def make_move(self, p0, p1):
        p0 = self._validate_pos(p0)
        p1 = self._validate_pos(p1)
        if not self._within_board(p0) or not self._within_board(p1):
            raise PositionError("p0 and p1 must be within board")
        team = self._whose_turn()
        piece = self._get_object(p0)
        if not isinstance(piece, Piece):
            raise PositionError('Only pieces can be moved')
        elif team != piece.team:
            raise TeamError("This piece doesn't belong to " + team)

        if piece._move_valid(p1):
            self._move_piece(p0, p1)
        else:
            raise PositionError('Piece cannot be moved to this position')
        self.turn += 1

    def _save_state(self):
        self.state =  {'turn': self.turn,
                       'board': self.board.copy(),
                       'objects': self.objects.copy()}
        for ob in self.objects:
            ob._save_state()

    def _reset_state(self):
        if self.state is None:
            raise StateError('No state has been saved for this object')
        for k, v in self.state.items():
            if k != 'states':
                setattr(self, k, v)
        for ob in self.objects:
            ob._reset_state()

    def _save_pos(self, pos):
        top = self._get_object(pos)
        if isinstance(top, Piece):
            top._save_state()
            top.on._save_state()
            self.pos_states[pos] = (top.on, top)
        else:
            top._save_state()
            self.pos_states[pos] = (top,)

    def _reset_pos(self, pos):
        if pos not in self.pos_states.keys():
            raise StateError('No state has been saved for this position')
        for ob in self.pos_states[pos]:
            ob._reset_state()
            self.board[pos] = ob.idx

    def _check_board_state(self):
        return True

    def _try_move(self, p0, p1):
        self._save_pos(p0)
        self._save_pos(p1)

        self._move_piece(p0, p1)
        self.turn += 1

        allowed = self._check_board_state()

        self.turn -= 1
        self._reset_pos(p0)
        self._reset_pos(p1)
        return allowed

    def try_move(self, p0, p1):
        p0 = self._validate_pos(p0)
        p1 = self._validate_pos(p1)
        if not self._within_board(p0) or not self._within_board(p1):
            raise PositionError("p0 and p1 must be within board")
        team = self._whose_turn()
        piece = self._get_object(p0)
        if not isinstance(piece, Piece):
            raise PositionError('Only pieces can be moved')
        elif team != piece.team:
            raise TeamError("This piece doesn't belong to " + team)
        if not piece._move_valid(p1):
            raise PositionError('Piece cannot be moved to this position')
        return self._try_move(p0, p1)

    def _list_piece_moves(self, piece, check_board=True):
        if isinstance(piece, int):
            piece = self.objects[piece]
        moves = []
        for m in piece._list_moves():
            if not check_board or self._try_move(piece.pos, m):
                moves.append((piece.pos, m))
        return moves

    def _list_pos_moves(self, pos):
        pos = self._validate_pos(pos)
        check = self.board[pos]
        return self._list_piece_moves(check)

    def _list_moves(self, check_board=True):
        team = self.teams[self.turn % self.nteams]
        moves = []
        for piece in self.pieces[team]:
            moves += self._list_piece_moves(piece, check_board=check_board)
        return moves
