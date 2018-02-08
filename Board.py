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
            raise UserWarning(
                "Trying to add a move that has already been played by player {} (x={} y={}".format(player, x, y))

        new_state = np.copy(self.moves_by[player][-1])
        new_state[x, y] = 1
        self.moves_by[player].append(new_state)

    def get_prev_moves(self, t=1, player=0):
        """Return numpy array of last t moves of player.

        :param t: How many most recent moves for the given player should be returned
        :returns: numpy array of shape (t, self.dimensions[0], self.dimensions[0]) where the
                  last element is the most recent move.
        """
        if t > len(self.moves_by[player]):
            raise UserWarning("Trying to get more(t={}) moves than have been played by player{} ({}).".format(
                t, player, len(self.moves_by[player])))
        return np.asarray(self.moves_by[player][-t:])
