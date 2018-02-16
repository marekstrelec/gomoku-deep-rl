from keras.layers import BatchNormalization, Conv2D, Dense, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2


class GomokuModel():
    def __init__(self, dimensions=(15, 15), t=1):
        self.dimensions = dimensions
        inputs = Input(shape=(dimensions[0], dimensions[1], 2 * t + 1))

        reg = l2(1e-4)

        # Convolutional layer
        conv_block = Conv2D(256, kernel_size=(3, 3), strides=1, kernel_regularizer=reg)(inputs)
        conv_block = BatchNormalization()(conv_block)
        conv_block = LeakyReLU(alpha=0.1)(conv_block)
        # AlphaGo Zero now repeats the above 19 or 39 times to form so-called 'residual tower'
        # AlphaLee Zero does something similar, but focuses more on convolution
        # TODO: Residal tower definition? How  deep?

        # Policy head
        ph = Conv2D(2, kernel_size=(1, 1), strides=1, kernel_regularizer=reg)(conv_block)
        ph = BatchNormalization()(ph)
        ph = LeakyReLU(alpha=0.1)(ph)
        policy_out = Dense(dimensions[0] * dimensions[1], kernel_regularizer=reg,
                           activation='softmax', name='policy_out')(ph)

        # Value head
        vh = Conv2D(1, kernel_size=(1, 1), strides=1, kernel_regularizer=reg)(conv_block)
        vh = BatchNormalization()(vh)
        vh = LeakyReLU(alpha=0.1)(vh)
        vh = Dense(256, kernel_regularizer=reg)(vh)
        vh = LeakyReLU(alpha=0.1)(vh)
        value_out = Dense(1, kernel_regularizer=reg, activation='tanh', name='value_out')(vh)

        self.model = Model(inputs=inputs, outputs=[policy_out, value_out])

        # Define losses for both outputs and the hyperparameters for SGD
        sgd = SGD(lr=0.01, decay=3 * 1e-6, momentum=0.9)
        loss = {'policy_out': 'binary_crossentropy', 'value_out': 'mean_squared_error'}
        loss_weights = {'policy_out': 1., 'value_out': 1}

        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=sgd)

    def train(self, generator, steps_per_epoch):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
                                 epochs=1, verbose=1, shuffle=False)
