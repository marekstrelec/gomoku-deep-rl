
import math

import numpy as np

from graph import Node


class MCTS(object):

    def __init__(self, board, model, player_positions, player, temperature=0.5):
        self.board = board
        self.model = model
        self.player_positions = player_positions
        self.temperature = temperature
        self.root_node = Node(None, player, None, 0)

    def compute_ucb1(self, node, c_ucb1=1 / (2 ** 1 / 2)):
        '''UCB1 algorithm

        Performs the UCB1 bandit learning algorith.

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
        :raises ZeroDivisionError: if UCB1 computed on unvisited nodes
        '''

        if node.N == 0:
            raise ZeroDivisionError("MCTS cannot compute UCB1 on unvisited nodes!")

        U = 2 * c_ucb1 * math.sqrt((2 * math.log(node.parent.N)) / node.N)
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
        ps, vs = self.model.predict(features)
        assert len(possible_actions) == len(ps) == len(vs)

        # add actions
        for idx in range(len(ps)):
            new_node = Node(selected_node, selected_node.player ==
                            0, possible_actions[idx], ps[idx])
            selected_node.add_child(new_node)

        return vs

    def backpropagate(self, node, value):
        '''Backpropagation

        The simulation result is backpropagated through the selected nodes to update their statistics.

        :param Node node: MCTS node containing a game position with relevant statistics
        :param int value: a scalar evaluation, estimating the probability of the current player winning from the position
        '''

        while not node.is_root():
            self.N += 1
            self.W += value
            self.Q = self.W / self.N
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

        # exponentiate
        pi_exp = [(node.N ** (1 / self.temperature), idx)
                  for idx, node in enumerate(root_children)]

        # normalize
        pi_denom = sum(pi_exp)
        pi_normalised = [n / pi_denom for n in pi_exp]

        # make the best move a root node
        best_move = root_children[max(pi_normalised)[1]]
        self.root_node = best_move
        best_move.parent = None

        return best_move

    def build_positions(self, node):
        '''Build a mosition from a node in the search tree

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
        merged = np.logical_or(positions[0], positions[1])
        r, c = np.where(merged == 0)
        return list(zip(r, c))

    def build_features(self, current_positions, possible_actions, player):
        """Builds features for NN

        :param list current_positions: a list containing positions for each player
        :param list possible_actions: a list of coordinates for all possible actions to play
        :return: features to be passed to the NN
        :rtype: numpy.ndarray
        """

        features = []
        for a in possible_actions:
            X = current_positions[player]
            X[a[0]][a[1]] = 1
            Y = current_positions[player == 0]
            C = self.board.get_player_mask(player)
            features.append(np.asarray([X, Y, C]))

        return np.asarray(features)

    def get_player_mask(self, player):
        """Makes a player mask

        Returns a boolean 2D array of the board shape representing a player.

        :param int player: player ID
        :return: a plane filled with 0s or 1s depending on the player to play
        :rtype: numpy.ndarray
        """

        return np.logical_or(np.zeros(self.dimensions, dtype=bool), player)
