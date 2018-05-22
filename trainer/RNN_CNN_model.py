from keras import Input, optimizers, losses, Model
from keras.activations import softmax
from keras.layers import Convolution2D, MaxPool2D, BatchNormalization, Activation, Flatten, Dense, GRU


def create_cnn_rnn_model(config):
    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    x = Convolution2D(16, (7, 7), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (5, 5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    # x = Flatten()(x)
    # x = Dense(64)(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # out = Dense(config.n_classes, activation=softmax)(x)

    num_channels = 32
    filter_W = 852
    filter_H = 8

    # InputLayer
    # inp = Input(tensor=x)

    x = GRU(500)(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = MaxPool2D()(x)
    out = Dense(config.n_classes, activation=softmax)(x)
    model = Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model
