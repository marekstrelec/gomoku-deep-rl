import unittest

import numpy as np
from data_store import SelfPlayHistory


class TestDataStore(unittest.TestCase):
    def setUp(self):
        self.dims = (15, 15)
        self.batch_size = 8
        self.n_data_to_store = 20
        self.data_store = SelfPlayHistory(
            n_data_to_store=self.n_data_to_store, batch_size=self.batch_size, board_dimensions=self.dims)

        def gen_dummy_data(n_dummy_data=10):
            dummy_board_states = np.random.randint(0, 2, (n_dummy_data, 3, *self.dims)) != 0
            dummy_move_probabilities = np.random.rand(n_dummy_data, *self.dims)
            dummy_values = np.random.rand(n_dummy_data)
            return dummy_board_states, dummy_move_probabilities, dummy_values

        self.gen_dummy_data = gen_dummy_data

    def test_empty_at_start(self):
        new_data_store = SelfPlayHistory()
        self.assertEqual(new_data_store.n_data, 0)

    def test_internal_structs_allocated_at_the_beginning(self):
        self.assertEqual(self.data_store.board_states.shape[0], self.n_data_to_store)
        self.assertEqual(self.data_store.pis.shape[0], self.n_data_to_store)
        self.assertEqual(self.data_store.zs.shape[0], self.n_data_to_store)

    def test_adding_less_data_than_capacity(self):
        n_some_data = 10
        some_data = self.gen_dummy_data(n_some_data)
        self.data_store.add_moves(*some_data)
        # Test size variable
        self.assertEqual(self.data_store.n_data, n_some_data)
        # Test copied data is identical
        self.assertTrue(np.all(self.data_store.board_states[:n_some_data] == some_data[0]))
        self.assertTrue(np.all(self.data_store.pis[:n_some_data] == some_data[1]))
        self.assertTrue(np.all(self.data_store.zs[:n_some_data] == some_data[2]))

        n_more_data = 5
        self.data_store.add_moves(*self.gen_dummy_data(n_more_data))
        # Test size variable
        self.assertEqual(self.data_store.n_data, n_some_data + n_more_data)

    def test_adding_data_of_uneven_lengths(self):
        n_dummy_data = 10
        dummy_data = self.gen_dummy_data(n_dummy_data)

        with self.assertRaises(AssertionError):
            self.data_store.add_moves(
                dummy_data[0][:n_dummy_data // 2], dummy_data[1], dummy_data[2])

        with self.assertRaises(AssertionError):
            self.data_store.add_moves(
                dummy_data[0], dummy_data[1][:n_dummy_data // 2], dummy_data[2])

        with self.assertRaises(AssertionError):
            self.data_store.add_moves(
                dummy_data[0], dummy_data[1], dummy_data[2][:n_dummy_data // 2])

    def test_adding_more_data_than_capacity(self):
        n_some_data = 15
        self.data_store.add_moves(*self.gen_dummy_data(n_some_data))

        n_more_data_than_capacity = 10
        self.data_store.add_moves(*self.gen_dummy_data(n_more_data_than_capacity))
        # More data than limit should overwrite oldest data and not add more data
        self.assertEqual(self.data_store.n_data, n_some_data)
        self.assertEqual(self.data_store.cursor, n_more_data_than_capacity)

        n_some_more_data = 7
        # Adding more data than n_data should work similarly
        self.data_store.add_moves(*self.gen_dummy_data(n_some_more_data))
        # More data than limit should overwrite oldest data and not add more data
        self.assertEqual(self.data_store.n_data, n_more_data_than_capacity + n_some_more_data)
        self.assertEqual(self.data_store.cursor, n_more_data_than_capacity + n_some_more_data)

    def test_reading_a_batch_returns_correct_num_samples(self):
        n_some_data = 15
        self.data_store.add_moves(*self.gen_dummy_data(n_some_data))
        self.assertEqual(self.batch_size, self.data_store.read_batch()[0].shape[0])

    def test_reading_a_batch_returns_different_results_every_time(self):
        n_some_data = 15
        self.data_store.add_moves(*self.gen_dummy_data(n_some_data))
        a_batch = self.data_store.read_batch()

        for i in range(0, 10):
            self.assertFalse(np.all([a_batch[0] == self.data_store.read_batch()[0]]))

    def test_is_generator(self):
        self.assertTrue(hasattr(self.data_store, '__iter__'))
        self.assertTrue(hasattr(self.data_store, '__next__'))


if __name__ == '__main__':
    unittest.main()