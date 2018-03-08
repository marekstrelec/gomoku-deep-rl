
import copy
import math

import numpy as np

from graph import Node


class MCTS(object):

    def __init__(self, board, model, player_positions, player, temperature=0.5):
        self.board = board
        self.model = model
        self.player_positions = player_positions
        self.temperature = temperature
        self.root_node = self.init_root(player_positions, player)

    def init_root(self, current_positions, player):

        def build_features(current_positions, player):
            features = np.zeros((1, 3, *self.board.dimensions))
            X = copy.deepcopy(current_positions[player])[0]
            Y = copy.deepcopy(current_positions[player == 0])[0]
            C = self.get_player_mask(player)

            X = X.reshape(1, *X.shape)
            Y = Y.reshape(1, *Y.shape)
            C = C.reshape(1, *C.shape)
            features[0] = np.vstack((X, Y, C))

            return features

        features = build_features(current_positions, player)
        ps, vs = self.model.model.predict(features)

        return Node(None, player, None, ps[0])

    def compute_ucb1(self, node, c_ucb1=1 / (2 ** 1 / 2)):
        '''PUCT algorithm

        Performs the PUCT bandit learning algorith.

        Addresses the exploration-exploitation dilemma in MCTS:
        every time a node (action) is to be selected within the existing tree, the choice
        may be modelled as an independent multi-armed bandit problem.
        The output is guaranteed to be within a constant factor of the best possible bound on
        the growth of regret.

        The value c_ucb1 = 1/√2 was shown by Kocsis and Szepesvari to satisfy
        the Hoeffding ineqality with rewards in the range [0, 1].
        (http://old.sztaki.hu/~szcsaba/papers/cg06-ext.pdf)

        :param Node node: the current child node
        :return: reward guess
        :rtype: int
        :raises ZeroDivisionError: if PUCT computed on unvisited nodes
        '''

        if node.N == 0:
            raise ZeroDivisionError("MCTS cannot compute PUCT on unvisited nodes!")

        siblings = node.parents.get_children()
        U = c_ucb1 * node.parent.P[node.move[0]][node.move[1]] * \
            math.sqrt(sum([n.N for n in siblings])) / (1 + node.N)
        return self.Q + U

    def selection(self):
        '''selection

        Starting at the root node, a child selection policy is recursively applied to descend
        through the tree until the most urgent expandable node is reached.

        The search control strategy used initially prefers actions with high prior probability and
        low visit count, but asympotically prefers actions with high action value

        :return: MCTS node to be furter expanded
        :rtype: Node
        '''

        current_node = self.root_node
        while not current_node.is_leaf():
            uct_values = [(self.compute_ucb1(node), idx)
                          for idx, node in enumerate(current_node.get_children())]
            current_node = current_node.get_children()[max(uct_values)[1]]

        return current_node

    def expansion(self, selected_node):
        '''Expansion

        Child nodes are added to expand the tree, according to the available actions.

        :param Node selected_node: MCTS node object selected for an expansion
        :return: a list of position evaluations from the NN
        :rtype: list
        '''

        # build features and send to the NN
        current_positions = self.build_positions(selected_node)
        possible_actions = self.get_all_possible_actions(current_positions)
        features = self.build_features(current_positions, possible_actions, selected_node.player)
        ps, vs = self.model.model.predict(features)
        # return ps, vs
        assert len(possible_actions) == len(ps) == len(vs)

        # add actions
        for idx in range(len(ps)):
            new_node = Node(selected_node, selected_node.player ==
                            0, possible_actions[idx], ps[idx])
            selected_node.add_child(new_node)

            self.backpropagate(new_node, vs[idx])

        # return vs

    def backpropagate(self, node, value):
        '''Backpropagation

        The simulation result is backpropagated through the selected nodes to update their statistics.

        :param Node node: MCTS node containing a game position with relevant statistics
        :param int value: a scalar evaluation, estimating the probability of the current player winning from the position
        '''

        while not node.is_root():
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def simulation(self):
        '''Simulation

        Selects a move a to play in the root position s, proportional to its exponentiated visit count,
        A parameters T is used, which controls the level of exploration.

        The subtree below the played mode is retained along with all its statistics, while the
        remainder of the tree is discarded. This is done by replacing the root_node with the played node.

        :return: the node containing a move to play
        :rtype: Node
        '''

        root_children = self.root_node.get_children()

        # normalize
        pi_exp = np.asarray([node.N for node in root_children])
        pi_exp = pi_exp ** (1 / self.temperature)
        pi_normalised = pi_exp / sum(pi_exp)

        # make the best move a root node
        best_move = root_children[pi_normalised.argmax()]
        self.root_node = best_move
        best_move.parent = None

        return best_move

    def build_positions(self, node):
        '''Build a position from a node in the search tree

        Player positions for the node are obtained by traversing the search tree backwards to
        the root and mergeing all encountered moves with the current player positions.

        :param Node node:
        :return: a list containing positions for each player
        :rtype: list
        '''
        player0 = np.copy(self.player_positions[0])
        player1 = np.copy(self.player_positions[1])
        while not node.is_root():
            if node.player == 0:
                player0[node.move[0]][node.move[1]] = 1
            elif node.player == 1:
                player1[node.move[0]][node.move[1]] = 1
            node = node.parent

        return [player0, player1]

    def get_all_possible_actions(self, positions):
        """Find all possible actions given a position

        : param position: positions of each player
        : returns: a list of coordinates of all unused positions
        """

        assert len(positions) == 2
        merged = np.logical_or(positions[0][0], positions[1][0])
        r, c = np.where(merged == 0)
        return list(zip(r, c))

    def build_features(self, current_positions, possible_actions, player):
        """Builds features for NN

        :param list current_positions: a list containing positions for each player
        :param list possible_actions: a list of coordinates for all possible actions to play
        :return: features to be passed to the NN
        :rtype: numpy.ndarray
        """

        features = np.zeros((len(possible_actions), 3, *self.board.dimensions))
        for idx, a in enumerate(possible_actions):
            X = copy.deepcopy(current_positions[player])[0]
            X[a[0]][a[1]] = 1
            Y = copy.deepcopy(current_positions[player == 0])[0]
            C = self.get_player_mask(player)

            X = X.reshape(1, *X.shape)
            Y = Y.reshape(1, *Y.shape)
            C = C.reshape(1, *C.shape)
            features[idx] = np.vstack((X, Y, C))

        return np.asarray(features)

    def get_player_mask(self, player):
        """Makes a player mask

        Returns a boolean 2D array of the board shape representing a player.

        :param int player: player ID
        :return: a plane filled with 0s or 1s depending on the player to play
        :rtype: numpy.ndarray
        """

        return np.logical_or(np.zeros(self.board.dimensions, dtype=bool), player)
