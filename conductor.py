import time
from ctypes import c_bool
from multiprocessing import Condition, Process
from multiprocessing.managers import BaseManager
from multiprocessing.sharedctypes import Value

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
train_monitor = Condition()

eval_monitor = Condition()
play_monitor = Condition()

stop = Value(c_bool, False)


def wait_for(monitor):
    monitor.acquire()
    monitor.wait()
    monitor.release()


def notify_all(monitor):
    monitor.acquire()
    monitor.notify_all()
    monitor.release()


def nn_handler(data_store, train_monitor, eval_monitor, stop_training):
    neural_net = GomokuModel()

    # Waits until players have produced enough data
    wait_for(train_monitor)
    print('NN - Enough data in data_store.')

    # Create a wrapper generator object
    class Generator(object):
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return self

        def __next__(self):
            return self.data.read_batch()

    while not stop_training.value:
        print('NN - Started training')
        # Trains NN for steps_per_epoch training steps
        neural_net.train(Generator(data_store), steps_per_epoch=3)
        # Saves model
        neural_net.save()

        # Signals start of evaluation
        notify_all(eval_monitor)
        print('NN - Epoch done. Notified evaluator')

        # Waits until evaluation is done
        wait_for(train_monitor)


def game_handler(data_store, train_monitor, play_monitor, enough_training_data, stop_playing):
    while not stop_playing.value:
        wait_for(play_monitor)
        print('GH - Playing now! (generating data)')
        # TODO: this will be real game class, with real players. right now just
        # generates dummy data into data_store

        for i in range(20):
            status = data_store.get_status()
            if status['n_data'] >= enough_training_data:
                notify_all(train_monitor)

            # Generates dummy moves
            dims = status['board_dimensions']
            n_dummy_data = 100

            dummy_board_states = np.random.randint(0, 2, (n_dummy_data, *dims, 3)) != 0
            dummy_move_probabilities = np.random.rand(n_dummy_data, *dims)
            dummy_values = np.random.rand(n_dummy_data)

            # Adds dummy moves to datastructure
            data_store.add_moves(dummy_board_states, dummy_move_probabilities, dummy_values)
            time.sleep(0.5)

        print('GH - Finished 25K games of self play')


def evaluation_handler(eval_monitor, play_monitor, stop_evaluator):
    while not stop_evaluator.value:
        # Wait for start of evaluation
        wait_for(eval_monitor)

        candidate = GomokuModel(load=True)
        # TODO: load best_model and play against candidate
        # then save winner as 'latest_model'

        print('EV - Running eval')
        time.sleep(3)

        print('EV - Finished evaluation. Notifying GH.')
        notify_all(play_monitor)


p_nn = Process(target=nn_handler, args=[shared_data, train_monitor, eval_monitor, stop])
p_nn.start()

p_game = Process(target=game_handler, args=[shared_data,
                                            train_monitor, play_monitor, enough_training_data, stop])
p_game.start()

p_eval = Process(target=evaluation_handler, args=[eval_monitor, play_monitor, stop])
p_eval.start()


cmd = None
while cmd != 'exit':
    cmd = input()
    if cmd == 'start':
        notify_all(play_monitor)
    elif cmd == 'exit':
        print('Inner exit condition')
        stop.value = True
        break

print('Exit from cmd loop. Waiting for processes to finish.')

p_game.join()
print('game Process Done')
p_nn.join()
print('nn Process Done')
p_game.join()
print('game Process Done')
