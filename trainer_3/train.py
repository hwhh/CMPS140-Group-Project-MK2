import multiprocessing
import os
import librosa.display
import librosa
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from trainer_3.FCN_Model import model_fn_aes, model_vgg, model_fn_vgg16_16s
from trainer_3.batch_generator import DataGenerator
from trainer_3.config import Config
from trainer_3.process_audio import prepare_data
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

"""
input should be in format :(X, 1, None, None, Somthing)
                (batch_size, chancels, None, length)

"""


def run(config):
    test = pd.read_csv("../input/sample_submission.csv")
    train = pd.read_csv("../input/train.csv")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    X_train, X_test, y_train, y_test = train_test_split(np.array(train.label_idx.keys()), train.label_idx.values,
                                                        test_size=0.2)
    partition = {'train': X_train, 'test': X_test}
    training_generator = DataGenerator(config, partition['train'], train.label_idx, config.train_dir, 0)
    test_generator = DataGenerator(config, partition['test'], train.label_idx, config.train_dir, 0)
    model = model_fn_aes(config)
    print(model.summary())
    model.fit_generator(generator=training_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        epochs=1)

    # x_train = prepare_data(train, config, '../input/audio_train_1/')
    # # x_test = prepare_data(train, config, '../input/audio_test/')
    #
    # y_train = to_categorical(train.label_idx, num_classes=config.n_classes)
    #
    # X_train, X_test, y_train, y_test = train_test_split(x_train, train.label_idx.values,
    #                                                     test_size=0.2)
    #
    # model = model_fn_aes(config)
    #
    # for seq, label in zip(X_train, y_train):
    #     model.train_on_batch(np.array(seq).reshape(1, 128, seq.shape[1], 1), [label])
    #
    #     callbacks_list = []
    #
    #     model.fit(X_train)
    #
    #     print()


def run_2(config):
    test = pd.read_csv("../input/sample_submission.csv")
    train = pd.read_csv("../input/train.csv")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])


    skf = StratifiedKFold(n_splits=config.n_folds).split(np.zeros(len(train)), train.label_idx)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()

        training_generator = DataGenerator(config, partition['train'], train.label_idx, config.train_dir, 0)
        test_generator = DataGenerator(config, partition['test'], train.label_idx, config.train_dir, 0)

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]

        checkpoint = ModelCheckpoint(config.job_dir + '/best_%d.h5' % i, monitor='val_loss', verbose=1,
                                     save_best_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold_%i' % i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]

        print('#' * 50)
        print('Fold: ', i)


#         curr_model = model_fn_basic(config)
#
#
# def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2, model_level, n_mfcc,
#                   audio_duration):
#     config = Config(sampling_rate=44100, audio_duration=audio_duration, n_folds=10,
#                     learning_rate=learning_rate, use_mfcc=True, n_mfcc=n_mfcc, train_csv=train_files,
#                     test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2,
#                     model_level=model_level, validated_labels_only=1)
#     run(config)
#
#
# create_config('../input/train.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train/',
#               '../input/audio_test/', 1, 40, 2)


plt.figure(figsize=(5, 4))
audio, _ = librosa.core.load('../input/audio_train/0aad0a16.wav')
S = librosa.feature.melspectrogram(y=audio)
librosa.display.specshow(librosa.power_to_db(S), x_axis='time', y_axis='mel')
plt.title('Gun Shot (DB Mel-Spectra)')
plt.show()
