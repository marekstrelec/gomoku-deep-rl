import time
from multiprocessing import Condition, Process
from multiprocessing.managers import BaseManager

import numpy as np
from data_store import SelfPlayHistory
from GomokuModel import GomokuModel

# Creates process-safe datastructure to store moves
BaseManager.register('SelfPlayHistory', SelfPlayHistory)
m = BaseManager()
m.start()
shared_data = m.SelfPlayHistory()

# Define multiprocessing condition, that indicates enough data has been
# accumulated to start training
enough_training_data = 10**3
start_training_condition = Condition()


def nn_handler(data_store, start_training):
    neural_net = GomokuModel()

    # Waits until players have produced enough data
    start_training.acquire()
    start_training.wait()
    start_training.release()
    print("Starting training!")

    # Create a wrapper generator object
    class Generator(object):
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return self

        def __next__(self):
            return self.data.read_batch()

    # Starts training. Returns after steps_per_epoch training steps
    neural_net.train(Generator(data_store), steps_per_epoch=10**3)

    # TODO: save model snapshot and trigger evaluation phase


def game_handler(data_store, start_training_condition):
    # TODO: this will be real game class, with real players. right now just
    # generates data into datastore

    for i in range(100):
        status = data_store.get_status()
        if status['n_data'] >= enough_training_data:
            start_training_condition.acquire()
            start_training_condition.notify_all()
            start_training_condition.release()

        # Generates dummy moves
        dims = status['board_dimensions']
        n_dummy_data = 100

        dummy_board_states = np.random.randint(0, 2, (n_dummy_data, *dims, 3)) != 0
        dummy_move_probabilities = np.random.rand(n_dummy_data, *dims)
        dummy_values = np.random.rand(n_dummy_data)

        # Adds dummy moves to datastructure
        data_store.add_moves(dummy_board_states, dummy_move_probabilities, dummy_values)
        time.sleep(0.5)


p_nn = Process(target=nn_handler, args=[shared_data, start_training_condition])
p_nn.start()

p_game = Process(target=game_handler, args=[shared_data, start_training_condition])
p_game.start()

p_game.join()
print('game Process Done')
p_nn.join()
print('nn Process Done')
