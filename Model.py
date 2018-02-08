from keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU


class Model():
    def __init__(self, dimensions=(20, 20), t=8):
        self.dimensions = dimensions
        inputs = Input(shape=(dimensions[0], dimensions[1], 2 * t + 1))

        l1 = Conv2D(256, kernel_size=(3, 3), strides=1)(inputs)
        l2 = BatchNormalization()(l1)
        l3 = LeakyReLU(alpha=0.1)(l2)
        # AlphaGo Zero now repeats the above 19 or 39 times to form so-called 'residual tower'
        # AlphaLee Zero does something similar, but focuses more on convolution
        # TODO: Residal tower definition? How deep?

        # Policy head
        ph = Conv2D(2, kernel_size=(1, 1), strides=1)(l3)
        ph = BatchNormalization()(ph)
        ph = LeakyReLU(alpha=0.1)(ph)
        policy_out = Dense(dimensions[0] * dimensions[1], activation='softmax')(ph)

        # Value head
        vh = Conv2D(1, kernel_size=(1, 1), strides=1)(l3)
        vh = BatchNormalization()(vh)
        vh = LeakyReLU(alpha=0.1)(vh)
        vh = Dense(256)(vh)
        vh = LeakyReLU(alpha=0.1)(vh)
        value_out = Dense(1, activation='tanh')(vh)

        self.model = Model(inputs=[inputs], outputs=[policy_out, value_out])

    def predict(self, input):
        return self.model.predict(input)
