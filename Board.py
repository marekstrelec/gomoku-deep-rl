from warnings import warn

import numpy as np


class Board(object):
    """ A simple class defining the Gomoku Board.
    State of the board is stored (together with the whole history of moves) in a list of boolean snapshots of the board after each player's move, where 1 marks player's moves and 0 marks both moves of the opponent and empty spaces.
    This unusual representation mirrors that of AlphaGo Zero, and should be convenient to feed to the RL-NN components. https://www.nature.com/articles/nature24270
    """

    def __init__(self, dimensions=(20, 20)):
        self.dimensions = dimensions
        self.empty_board = np.zeros(dimensions, dtype=bool)
        self.moves_by = [[], []]
        self.moves_by[0].append(self.empty_board)
        self.moves_by[1].append(self.empty_board)

    def add_move(self, x, y, player=0):
        """Records a new move at position x,y for player."""

        if self.moves_by[player][-1][x, y] != 0:
            warn("Trying to add a move that has already been played by player {} (x={} y={})".format(player, x, y))

        new_state = np.copy(self.moves_by[player][-1])
        new_state[x, y] = 1
        self.moves_by[player].append(new_state)

    def get_prev_moves(self, n=1, player=0):
        """Returns numpy array of last t moves of player.

        :param n: How many most recent moves for the given player should be returned. If the player haven't played that many moves, empty board will be returned.
        :returns: numpy array of shape (t, self.dimensions[0], self.dimensions[0]) where the
                  last element is the most recent move.
        """
        moves_played = len(self.moves_by[player])
        if n > moves_played:
            blanks = np.zeros((n - moves_played, *self.dimensions), dtype=bool)
            return np.vstack((blanks, self.moves_by[player]))
        return np.asarray(self.moves_by[player][-n:])

    def get_features(self, n=1):
        """Compiles an array of feature planes used as input into the NN.

        :param n: How many most recent pairs of moves should be returned
        :returns: numpy array of t*2 + 1 feature planes [X-1, Y-1, X-2, Y-2, ..., X-t, Y-t, C] where X-1 is
                  the last move that was played, and Y-1 was last move played by the other player.
        """
        raise NotImplementedError

    def get_player_board(self, player=0):
        """Returns a boolean 2D array of board shape"""
        return np.logical_or(np.zeros(self.dimensions, dtype=bool), player)
