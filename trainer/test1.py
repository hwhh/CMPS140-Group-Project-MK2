"""



• C1P1 (Convolution followed by Pooling) – 96 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C2P2 (Convolution followed by Pooling) – 128 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C3P3 (Convolution followed by Pooling) – 128 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C4P4 (Convolution followed by Pooling) – 256 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C5P5 (Convolution followed by Pooling) – 256 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)




• FCv1 (Convolution Layer) - 256 Filter of receptive fields (4 x 4), Stride - 1, No padding and ReLU activation.
• FCv2 (Convolution Layer) - nclasses Filters of size (1 X1), Stride - 1, and Sigmoid Activation.


"""
from keras import Input, Model, losses
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPool2D, MaxPooling2D, Conv2D, Conv1D, Dropout
from keras.optimizers import Adam


def model_fn_aes(config):
    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    x = Conv1D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Dropout(0.5)(x)
    x = Conv2D(256, kernel_size=(4, 4), strides=(1, 1), padding="none", activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Conv2D(config.n_classes, kernel_size=(1, 1), strides=(1, 1), padding="none", activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

