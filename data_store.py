import numpy as np


class SelfPlayHistory(object):
    """A class to store all moves, their actual probabilities and the end of the game they led to (win/loss). Shall be instanciated using Python's multiprocessing managers, which should add all necessary locks&semaphores.
    Internally, creates fixed size (n_data_to_store items) numpy arrays to store the data. Once the capacity is full, it starts overwriting items from the beginning (sort of a FILO buffer).

    An instance of this class can be passed as a generator, as it contains the __next__() and __iter__() methods.
    """

    def __init__(self, n_data_to_store=10**4, batch_size=100, board_dimensions=(19, 19), n_moves_in_state=3):
        self.batch_size = batch_size
        self.board_dimensions = board_dimensions
        self.capacity = n_data_to_store
        self.cursor = 0
        self.n_data = 0
        self.board_states = np.zeros(
            (n_data_to_store, board_dimensions[0], board_dimensions[1], n_moves_in_state), dtype=bool)
        self.pis = np.zeros((n_data_to_store, board_dimensions[0], board_dimensions[1]))
        self.zs = np.zeros((n_data_to_store))

    def add_moves(self, board_states, move_probabilities, values):
        """Adds board_states.shape[0] items into the internal numpy data_stores. Asserts same-length data and does nothing if the arguments are of length 0.
        Note that if board_states.shape[0] is greater than the unallocated space in the array, it just restarts the counter and starts overwriting the data from beginning (this operates under the assumption that data will be added in small batches, and the overall size of the data store will be orders of magnitute larger.)
        """
        n_new_data = board_states.shape[0]
        assert move_probabilities.shape[0] == n_new_data and values.shape[0] == n_new_data
        if n_new_data is 0:
            return
        new_cursor = self.cursor + n_new_data

        # Assuming self.capacity is big and there aren't many board states added,
        # just ignore the last bit of allocated space, and start overwriting data
        # from the beginning
        if new_cursor >= self.capacity:
            self.cursor = 0
            new_cursor = n_new_data

        self.board_states[self.cursor:new_cursor] = board_states
        self.pis[self.cursor:new_cursor] = move_probabilities
        self.zs[self.cursor:new_cursor] = values

        self.cursor = new_cursor
        if self.cursor > self.n_data:
            self.n_data = self.cursor

    def read_batch(self):
        """Returns a random batch of data. We sample with repetition.
        TODO: should we skew this towards newest moves?
        """
        indices = np.random.randint(0, self.n_data, self.batch_size)
        flat_policy = self.pis[indices].reshape(indices.shape[0], -1)
        return (self.board_states[indices], {'policy_out': flat_policy, 'value_out': self.zs[indices]})

    def get_status(self):
        return {'cursor': self.cursor,
                'n_data': self.n_data,
                'capacity': self.capacity,
                'batch_size': self.batch_size,
                'board_dimensions': self.board_dimensions
                }
