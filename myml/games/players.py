import re


class Player(GameObject):

    def _set_game(self, game):
        super()._set_game(game)
        self.idx = len(game.players)
        if not self.name:
            self.name = 'Player {0}'.format(self.idx + 1)

    def _str_to_pos(self, pos):
        match = re.fullmatch(r'(?P<l>[a-z]+)(?P<n>[0-9]+)', pos, re.I)
        if match is None:
            match = re.fullmatch(r'(?P<n>[0-9]+)(?P<l>[a-z]+)', pos, re.I)
        if match is None or len(match.groups()) != 2:
            raise PositionError('The position should be formatted A1 or 1A')
        pos = (int(match.group('n')),
               int(_letters_to_number(match.group('l'))))
        return pos

    def _get_move(self):
        print(self.game.board)
        m0 = input('Select the piece to be moved:')
        m0 = self._str_to_pos(m0)
        m1 = input('Move piece to:')
        m1 = self._str_to_pos(m1)
        return m0, m1
