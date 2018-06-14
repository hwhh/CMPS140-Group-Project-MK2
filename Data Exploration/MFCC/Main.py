import os
import shutil

import librosa
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import (Convolution2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, Activation, K)
from keras.utils import to_categorical
import pandas as pd

from keras import Input, models, optimizers, losses
from keras.activations import softmax
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold

from MFCC.Config import Config

test = pd.read_csv("../input/sample_submission.csv")
train = pd.read_csv("../input/train.csv")
config = Config(sampling_rate=44100, audio_duration=2, n_folds=10,
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}

train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])


# The layers of the nureal network
def get_2d_conv_model(config):
    nclass = config.n_classes

    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X


def normalize_data(set):
    mean = np.mean(set, axis=0)
    std = np.std(set, axis=0)
    return (set - mean) / std


def run():
    get_2d_conv_model(config)
    X_train = prepare_data(train, config, '../input/audio_train/')
    X_test = prepare_data(test, config, '../input/audio_test/')
    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    PREDICTION_FOLDER = "predictions_2d_conv"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)

    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]

        checkpoint = ModelCheckpoint('best_%d.h5' % i, monitor='val_loss', verbose=1, save_best_only=True)

        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i' % i, write_graph=True)

        callbacks_list = [checkpoint, early, tb]

        print("#" * 50)
        print("Fold: ", i)

        model = get_2d_conv_model(config)

        history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list,
                            batch_size=64, epochs=config.max_epochs)
        model.load_weights('best_%d.h5' % i)

        # Save train predictionsfc
        predictions = model.predict(X_train, batch_size=64, verbose=1)
        np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy" % i, predictions)

        # Save test predictions
        predictions = model.predict(X_test, batch_size=64, verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy" % i, predictions)

        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv" % i)


def create_predictions():
    pred_list = []
    for i in range(10):
        pred_list.append(np.load("./predictions_2d_conv/test_predictions_%d.npy" % i))
    for i in range(10):
        pred_list.append(np.load("./predictions_2d_conv/test_predictions_%d.npy" % i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction * pred
    prediction = prediction ** (1. / len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('../input/sample_submission.csv')
    test['label'] = predicted_labels
    test[['fname', 'label']].to_csv("1d_2d_ensembled_submission.csv", index=False)


run()
