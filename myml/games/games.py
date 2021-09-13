import numpy as np


class Game(object):

    def __init__(self, name='Game'):
        if not isinstance(name, str):
            raise TypeError('"name" must be a string"')
        self.name = name
        self.board = None
        self.players = []
        self.t2p = {}

    def add_board(self, board):
        board._set_game(self)
        self.board = board

    def add_player(self, player, team=None):
        if self.board is None:
            raise ParentError('You must add a board before adding players')
        player._set_game(self)
        if len(self.t2p.keys()) >= self.board.nteams:
            raise TeamError('All teams have already been taken by players')
        elif team is None:
            team = [t for t in self.board.teams if t not in self.t2p.keys()][:1]
        elif not isinstance(team, list):
            team = [team]
        for t in team:
            t = self.board._validate_team(t)
            if t in self.t2p.keys():
                raise TeamError('{0} already has a player to control it')
            else:
                self.t2p[t] = player
        self.players.append(player)

    def _play_turn(self):
        turn = self.board.turn
        player = self.t2p[self.board._whose_turn()]
        print("It is {0}'s turn".format(player))
        while self.board.turn == turn:
            m0, m1 = player._get_move()
            try:
                possible = self.board.try_move(m0, m1)
            except Exception as e:
                possible = False
                print(e)
            if possible:
                self.board.make_move(m0, m1)
            else:
                print('This move is not possible, try again.')


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


class BoardObject(object):

    def __init__(self, name='Atom'):
        if not isinstance(name, str):
            raise TypeError('"name" must be a string"')
        self.name = name
        self.board = None
        self.idx = None
        self.team = None
        self.pos = None
        self.state = None

    def __repr__(self):
        return ' '

    def _set_board(self, board):
        if self.board:
            raise ParentError('This is already part of ' + self.board.name)
        else:
            self.board = board

    def _set_team(self, team):
        if team is not None:
            raise ValueError('This object cannot be given a team')

    def _set_pos(self, pos):
        pass

    def _save_state(self):
        self.state = {}

    def _reset_state(self):
        if self.state is None:
            raise StateError('No state has been saved for this object')
        for k, v in self.state.items():
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

    def __repr__(self):
        return 't{0}'.format(self.board.teams.index(self.team))


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

    def __repr__(self):
        return 't{0}'.format(self.board.teams.index(self.team))

    def _set_pos(self, pos):
        if self.pos:
            raise PositionError("This piece's position has already been set")
        check = self.board._get_object(pos)
        if check._can_set_on(self):
            self.pos = tuple(pos)
            check._set_on(self)
        else:
            raise PositionError('This position is already in use')
        self.on = check

    def _save_state(self):
        self.state =  {'pos': self.pos,
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
        check = self.board._get_object(target)
        return check._can_land_on(self)

    def _update_pos(self, pos):
        in_pos = self.board._get_object(pos)
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
