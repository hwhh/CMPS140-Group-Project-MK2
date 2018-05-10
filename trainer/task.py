import argparse
import os
import shutil

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from trainer.config import Config
from trainer.model import *

from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

PREDICTION_FOLDER = "predictions_2d_conv"


def run(config):
    file_stream = file_io.FileIO(config.test_csv[0], mode='r')
    test = pd.read_csv(StringIO(file_stream.read()))

    file_stream = file_io.FileIO(config.train_csv[0], mode='r')
    train = pd.read_csv(StringIO(file_stream.read()))

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    model_fn(config)
    X_train = prepare_data(train, config, config.train_dir)
    X_test = prepare_data(test, config, config.test_dir)
    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    try:
        os.makedirs(config.job_dir)
    except:
        pass

    skf = StratifiedKFold(n_splits=config.n_folds).split(np.zeros(len(train)), train.label_idx)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]

        checkpoint = ModelCheckpoint(config.job_dir + '/best_%d.h5' % i, monitor='val_loss', verbose=1,
                                     save_best_only=True)

        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)

        tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold_%i' % i, write_graph=True)

        callbacks_list = [checkpoint, early, tb]

        print('#' * 50)
        print('Fold: ', i)

        curr_model = model_fn(config)

        curr_model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list,
                       batch_size=64, epochs=config.max_epochs)

        curr_model.load_weights(config.job_dir + '/best_%d.h5' % i)

        if config.job_dir.startswith('gs://'):
            curr_model.save('model_%d.h5' % i)
            copy_file_to_gcs(config.job_dir, 'model_%d.h5' % i)
        else:
            curr_model.save(os.path.join(config.job_dir, 'model_%d.h5' % i))

        predictions = curr_model.predict(X_train, batch_size=64, verbose=1)
        np.save('train_predictions_%d.npy' % i, predictions)
        copy_file_to_gcs(config.job_dir, 'train_predictions_%d.npy' % i)

        # Save test predictions
        predictions = curr_model.predict(X_test, batch_size=64, verbose=1)
        np.save('test_predictions_%d.npy' % i, predictions)
        copy_file_to_gcs(config.job_dir, 'test_predictions_%d.npy' % i)

        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv('predictions_%d.csv' % i)
        copy_file_to_gcs(config.job_dir, 'predictions_%d.csv' % i)

        # Convert the Keras model to TensorFlow SavedModel
        to_savedmodel(curr_model, os.path.join(config.job_dir, 'export_%d' % i))


def create_predictions(config):
    file_stream = file_io.FileIO(config.train_csv[0], mode='r')
    train = pd.read_csv(StringIO(file_stream.read()))

    LABELS = list(train.label.unique())

    pred_list = []

    for i in range(config.n_folds):
        with file_io.FileIO(config.job_dir + '/test_predictions_%d.npy' % i, mode='r') as input_f:
            pred_list.append(np.load(input_f))

    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction * pred
    prediction = prediction ** (1. / len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]

    file_stream = file_io.FileIO(config.test_csv[0], mode='r')
    test = pd.read_csv(StringIO(file_stream.read()))

    test['label'] = predicted_labels
    test[['fname', 'label']].to_csv('1d_2d_ensembled_submission.csv', index=False)
    copy_file_to_gcs(config.job_dir, '1d_2d_ensembled_submission.csv')


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2):
    config = Config(sampling_rate=44100, audio_duration=2, n_folds=10,
                    learning_rate=learning_rate, use_mfcc=True, n_mfcc=40, train_csv=train_files,
                    test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2
                    )
    run(config)
    create_predictions(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--user_arg_1',
                        required=True,
                        type=str,
                        help='Directory of audio train files '
                        )
    parser.add_argument('--user_arg_2',
                        required=True,
                        type=str,
                        help='Directory of audio test files '
                        )

    parse_args, unknown = parser.parse_known_args()
    create_config(**parse_args.__dict__)

# gcloud ml-engine jobs submit training $JOB_NAME \
#     --job-dir $OUTPUT_PATH \
#     --runtime-version 1.4 \
#     --module-name trainer.task \
#     --package-path trainer/ \
#     --region $REGION \
#     --scale-tier STANDARD_1 \
#     -- \
#     --train-files $TRAIN_DATA \
#     --eval-files $EVAL_DATA \
#     --user_arg_1 $TRAIN_DIR \
#     --user_arg_2 $TEST_DIR \
#     --learning-rate 0.001 \
#     --verbosity DEBUG
