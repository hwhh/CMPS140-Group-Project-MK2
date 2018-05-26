# coding=utf-8
"""
• C1P1 (Convolution followed by Pooling) – 96 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C2P2 (Convolution followed by Pooling) – 128 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C3P3 (Convolution followed by Pooling) – 128 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C4P4 (Convolution followed by Pooling) – 256 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• C5P5 (Convolution followed by Pooling) – 256 Filters of receptive field (3 X 3), Stride - 1, Padding - 1 and ReLU activation. Max pooling over (2X2)
• FCv1 (Convolution Layer) - 256 Filter of receptive fields (4 x 4), Stride - 1, No padding and ReLU activation.
• FCv2 (Convolution Layer) - nclasses Filters of size (1 X1), Stride - 1, and Sigmoid Activation.

"""

from keras import Input, Model, losses, Sequential
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D, Convolution2D, Activation, Dense
from keras.layers import MaxPooling2D
from keras.optimizers import Adam

from trainer_2.models.BilinearUpSampling2D import BilinearUpSampling2D
from trainer_3.SpatialPyramidPooling import SpatialPyramidPooling


def model_fn_aes(config):
    inp = Input(shape=(None, None, 1))
    # Block 1
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', name='block1_conv1')(inp)

    # inp = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(None, None, 1))
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
    x = Conv2D(256, kernel_size=(4, 4), strides=(1, 1), padding="valid", activation='relu')(x)

    x = Dropout(0.5)(x)
    x = Conv2D(config.n_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation='sigmoid')(x)

    x = GlobalAveragePooling2D(data_format='channels_last')(x)

    model = Model(inputs=inp, outputs=x)
    opt = Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model


def model_fn_vgg16_16s(config):
    # input_shape = (config.dim[0], config.dim[1], 1)
    # image_size = input_shape[0:2]

    inp = Input(shape=(None, None, 1))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inp)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2), name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(0.5)(x)

    # classifying layer
    x = Conv2D(41, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid',
               strides=(1, 1))(x)

    # x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(inp, x)
    opt = Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def model_vgg(config):
    batch_size = 64
    num_channels = 3
    num_classes = 10

    model = Sequential()

    # uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, None, None)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model.compile(loss='categorical_crossentropy', optimizer='sgd')
