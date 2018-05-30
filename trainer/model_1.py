from keras import backend as K, Input, models, losses, optimizers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense


def model_fn_vnn(config, nb_layers=4):
    K.set_image_data_format('channels_last')  # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last
    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5  # conv. layer dropout
    dl_dropout = 0.6  # dense layer dropout

    input_shape = (128, 41, 2)
    inp = Input(shape=input_shape)
    x = Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape)(inp)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    for layer in range(nb_layers - 1):  # add more layers than just the first
        x = Conv2D(nb_filters, kernel_size)(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(cl_dropout)(x)

    x = Flatten()(x)
    x = Dense(128)(x)  # 128 is 'arbitrary' for now
    x = Activation('elu')(x)
    x = Dropout(dl_dropout)(x)
    x = Dense(config.n_classes)(x)
    x = Activation("softmax")(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model
