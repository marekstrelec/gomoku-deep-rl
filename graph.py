
class Node(object):

    '''MCTS node

    Each node s in the search tree contains edges (s, a) for all legal actions a ∈ A(s).
    Each edge stores a set of statistics:
        - N: the visit count
        - W: the total action value
        - Q: the mean action value
        - P: the prior probability of selecting that edge

    The value information for the positions encountered during search are updated to predict the
    winner of simulated games of self­play.

    Each node also contains a move information to represent a possible action. The relevant
    position can be obtained by traversing backwards to the root and mergeing all encountered moves
    with a starting player position.
    '''

    def __init__(self, parent, player, move, pa):
        assert isinstance(parent, Node) or parent is None
        assert player in [0, 1]
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert isinstance(pa, float)

        self.parent = parent
        self.children = None
        self.player = player
        self.move = move
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = pa

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children is None

    def get_children(self):
        return self.children

    def add_child(self, child):
        if self.children is None:
            self.children = []

        self.children.append(child)
