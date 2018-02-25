
import numpy as np


class IllegalMove(Exception):

    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return repr("Player {0} cannot put a stone on the possition {1} {2}!".format(self.player, self.x, self.y))


class Board(object):
    """ A simple class defining the Gomoku Board.
    State of the board is stored (together with the whole history of moves) in a list of boolean
    snapshots of the board after each player's move, where 1 marks player's moves and 0 marks both moves of the opponent and empty spaces.
    This unusual representation mirrors that of AlphaGo Zero, and should be convenient to feed to the RL-NN components.
    https://www.nature.com/articles/nature24270
    """

    def __init__(self, dimensions=(15, 15)):
        self.dimensions = dimensions
        self.moves_by = [[], []]
        self.moves_by[0].append(np.zeros(dimensions, dtype=bool))
        self.moves_by[1].append(np.zeros(dimensions, dtype=bool))

    def add_move(self, player, x, y):
        """Records a new move at position x,y for player."""

        # check for illegal moves
        if x < 0 or x >= self.moves_by[0][-1].shape[0]:
            raise IllegalMove(player, x, y)
        elif y < 0 or y >= self.moves_by[0][-1].shape[1]:
            raise IllegalMove(player, x, y)
        elif self.moves_by[0][-1][x, y] or self.moves_by[1][-1][x, y]:
            raise IllegalMove(player, x, y)

        # copy the last position and put a stone
        new_state = np.copy(self.moves_by[player][-1])
        new_state[x, y] = 1
        self.moves_by[player].append(new_state)

    get_prev_moves(self, n=1, player=0):
        """Returns numpy array of last t moves of player.

        :param n: How many most recent moves for the given player should be returned. If the player haven't played that many moves, the return array will be padded with a zeroed out (empty) board.
        :returns: numpy array of shape (t, self.dimensions[0], self.dimensions[0]) where the
                  last element is the most recent move.
        """
        moves_played = len(self.moves_by[player])
        if n > moves_played:
            blanks = np.zeros((n - moves_played, *self.dimensions), dtype=bool)
            return np.vstack((blanks, self.moves_by[player]))
        return np.asarray(self.moves_by[player][-n:])

    def get_features(self, n=1, player=0):
        """Compiles an array of feature planes used as input into the NN.

        :param n: How many most recent pairs of moves should be returned
        :returns: numpy array of n*2 + 1 feature planes [X-1, Y-1, X-2, Y-2, ..., X-n, Y-n, C] where X-1 is
                  the last move that was played by player, and Y-1 was last move played by the other player. C is a mask: plane filled with 0 or 1 depending on the player's colour.
        """
        p_features = self.get_prev_moves(n, player)
        otherp_features = self.get_prev_moves(n, 1 - player)
        features = []
        for i in range(p_features.shape[0] - 1, -1, -1):
            features.append(p_features[i])
            features.append(otherp_features[i])
        features.append(self.get_player_board(player))
        return np.asarray(features)

    def get_player_board(self, player=0):
        """Returns a boolean 2D array of board shape"""
        return np.logical_or(np.zeros(self.dimensions, dtype=bool), player)
