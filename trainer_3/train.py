import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from trainer_3.FCN_Model import model_fn_aes, model_vgg, model_fn_vgg16_16s
from trainer_3.config import Config
from trainer_3.process_audio import prepare_data
import numpy as np
from keras import backend as K

"""
input should be in format :(X, 1, None, None, Somthing)
                (batch_size, chancels, None, length)

"""


def run(config):
    test = pd.read_csv("../input/sample_submission.csv")
    train = pd.read_csv("../input/train_1.csv")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    x_train = prepare_data(train, config, '../input/audio_train_1/')
    # x_test = prepare_data(train, config, '../input/audio_test/')

    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

    skf = StratifiedKFold(n_splits=2).split(np.zeros(len(train)), train.label_idx)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        x, y, x_val, y_val = x_train[train_split], y_train[train_split], np.array(x_train[val_split]), y_train[val_split]

        checkpoint = ModelCheckpoint(config.job_dir + '/best_%d.h5' % i, monitor='val_loss', verbose=1,
                                     save_best_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        # tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold_%i' % i, write_graph=True)
        callbacks_list = [checkpoint, early]

        print(('Fold: %d' % i) + '\n' + '#' * 50)

        curr_model = model_fn_aes(config)

        curr_model.fit(x, y, validation_data=(x_val, y_val), callbacks=callbacks_list,
                       batch_size=64, epochs=config.max_epochs)

        curr_model.load_weights(config.job_dir + '/best_%d.h5' % i)


def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2, model_level, n_mfcc,
                  audio_duration):
    config = Config(sampling_rate=44100, audio_duration=audio_duration, n_folds=10,
                    learning_rate=learning_rate, use_mfcc=True, n_mfcc=n_mfcc, train_csv=train_files,
                    test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2,
                    model_level=model_level, validated_labels_only=1)
    run(config)


create_config('../input/train.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train/',
              '../input/audio_test/', 1, 40, 2)
