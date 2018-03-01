
import numpy as np


class IllegalMove(Exception):

    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return repr("Player {0} cannot put a stone on the possition {1} {2}!".format(self.player, self.x, self.y))


class IllegalAction(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class Board(object):
    """ A simple class defining the Gomoku Board.

    State of the board is stored (together with the whole history of moves) in a list of boolean
    snapshots of the board after each player's move, where 1 marks player's moves and 0 marks both moves of the opponent and empty spaces.
    This unusual representation mirrors that of AlphaGo Zero, and should be convenient to feed to the RL-NN components.
    https://www.nature.com/articles/nature24270
    """

    def __init__(self, dimensions=(15, 15)):
        self.dimensions = dimensions

        self.names = {
            0: 'White',
            1: 'Black'
        }

        self.positions = []
        self.positions.append(np.array([np.zeros(dimensions, dtype=bool), 1]))
        self.positions.append(np.array([np.zeros(dimensions, dtype=bool), 0]))

    def add_move(self, player, x, y):
        """Records a new move at position x,y for player."""

        # check for a correct player order
        if self.positions[-1][1] == player:
            raise IllegalAction("Player {0} cannot play twice in a row!")

        # check for illegal moves
        if x < 0 or x >= self.dimensions[0]:
            raise IllegalMove(player, x, y)
        elif y < 0 or y >= self.dimensions[1]:
            raise IllegalMove(player, x, y)
        elif self.positions[-1][0][x, y] or self.positions[-2][0][x, y]:
            raise IllegalMove(player, x, y)

        # copy the last position and put a stone
        new_state = np.copy(self.positions[-1][0])
        new_state[x, y] = 1
        self.positions.append(np.array([new_state, player]))
