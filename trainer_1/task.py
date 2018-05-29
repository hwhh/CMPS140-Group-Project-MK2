import argparse

import pandas as pd
from imageio import imread
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from pandas.compat import StringIO
from sklearn.model_selection import StratifiedKFold

from trainer_2.config import Config
from trainer_2.utils import *
from trainer_1.model_1 import model_fn_vnn

PREDICTION_FOLDER = "predictions_2d_conv"


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 models into TensorFlow SavedModel."""
    builder = saved_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def is_file_available(filepath):
    try:
        return stat(filepath)
    except NotFoundError as e:
        return False


def load_melgram(file_path, config):
    # auto-detect load method based on filename extension
    name, extension = os.path.splitext(file_path)
    if '.npy' == extension:
        if config.job_dir.startswith('gs://') and is_file_available(file_path):
            with file_io.FileIO(file_path, mode='r') as input_f:
                return np.load(input_f)
        elif not config.job_dir.startswith('gs://') and os.path.exists(file_path):
            return np.load(file_path)
    elif '.npz' == extension:  # compressed npz file (preferred)
        with np.load(file_path) as data:
            return data['melgram']
    elif ('.png' == extension) or ('.jpeg' == extension):
        arr = imread(file_path)
        melgram = np.reshape(arr, (1, 1, arr.shape[0], arr.shape[1]))  # convert 2-d image
        return np.flip(melgram, 0)  # we save images 'rightside up' but librosa internally presents them 'upside down'
    return


def shuffle_XY_paths(X, Y, paths):  # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0])
    # print("shuffle_XY_paths: Y.shape[0], len(paths) = ",Y.shape[0], len(paths))
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths[:]
    for i in range(len(idx)):
        newX[i] = X[idx[i], :, :]
        newY[i] = Y[idx[i], :]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths


def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None


def build_data_set(config, path, load_frac=1.0, batch_size=None, tile=False):
    if config.job_dir.startswith('gs://'):
        file_stream = file_io.FileIO(config.test_csv[0], mode='r')
        test = pd.read_csv(StringIO(file_stream.read()))
        file_stream = file_io.FileIO(config.train_csv[0], mode='r')
        train = pd.read_csv(StringIO(file_stream.read()))
    else:
        test = pd.read_csv("../input/sample_submission.csv")
        train = pd.read_csv("../input/train.csv")

    labels = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(labels)}
    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])
    category_group = train.groupby(['label'])
    total_load = int(len(train) * load_frac)

    X = np.empty(shape=(total_load, config.dim[0], config.dim[1], config.dim[2]))
    Y = np.zeros((total_load, config.n_classes))

    paths = []

    load_count = 0
    for idx, label in enumerate(labels):
        this_Y = np.array(encode_class(label, labels))
        this_Y = this_Y[np.newaxis, :]
        class_files = category_group.groups[label]  # Get the files for specific class
        n_files = len(class_files)
        n_load = int(n_files * load_frac)  # n_load is how many files of THIS CLASS are expected to be loaded
        file_list = class_files[0:n_load]
        for idx2, infilename in enumerate(file_list):  # Load files in a particular class
            audio_path = path + '/' + infilename.replace(".wav", ".npy")
            X[load_count, :, :] = load_melgram(audio_path, config)
            Y[load_count, :] = this_Y
            paths.append(audio_path)
            load_count += 1
            if load_count >= total_load:  # Abort loading files after last even multiple of batch size
                break
        if load_count >= total_load:  # Second break needed to get out of loop over classes
            break
    if load_count != total_load:  # check to make sure we loaded everything we thought we would
        raise Exception("Loaded " + str(load_count) + " files but was expecting " + str(total_load))

    X, Y, paths = shuffle_XY_paths(X, Y, paths)  # mix up classes, & files within classes

    return X, Y, paths, labels


def run(config, epochs=50, batch_size=20, val_split=0.25, tile=False):
    if config.job_dir.startswith('gs://'):
        file_stream = file_io.FileIO(config.test_csv[0], mode='r')
        test = pd.read_csv(StringIO(file_stream.read()))
        file_stream = file_io.FileIO(config.train_csv[0], mode='r')
        train = pd.read_csv(StringIO(file_stream.read()))
        path = 'gs://cmps140-205021-mlengine/input_preprocessed/audio_train'
    else:
        test = pd.read_csv("../input/sample_submission.csv")
        train = pd.read_csv("../input/train.csv")
        path = '../out/audio_train'

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    np.random.seed(1)

    # Get the data
    X_train, Y_train, paths_train, class_names = build_data_set(config, path=path,
                                                                batch_size=batch_size,
                                                                tile=tile)
    # X_test, Y_test, paths_test, class_names = build_data_set(config, path='../out/audio_test',
    #                                                          batch_size=batch_size,
    #                                                          tile=tile)
    # Instantiate the model

    try:
        os.makedirs(config.job_dir)
    except OSError:
        pass

    skf = StratifiedKFold(n_splits=config.n_folds).split(np.zeros(len(train)), train.label_idx)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        X, y, X_val, y_val = X_train[train_split], Y_train[train_split], X_train[val_split], Y_train[val_split]

        checkpoint = ModelCheckpoint(config.job_dir + '/best_%d.h5' % i, monitor='val_loss', verbose=1,
                                     save_best_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=12)

        tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold_%i' % i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]

        print('#' * 50)
        print('Fold: ', i)

        curr_model = model_fn_vnn(X_train.shape, config)

        # curr_model.fit(X, y_train, batch_size=1)
        # curr_model = model_fn_vgg16_16s(config)
        # curr_model.fit(X_train, y_train, batch_size=1)

        curr_model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, batch_size=64,
                       epochs=config.max_epochs)

        curr_model.load_weights(config.job_dir + '/best_%d.h5' % i)

        if config.job_dir.startswith('gs://'):
            curr_model.save('model_%d.h5' % i)
            copy_file_to_gcs(config.job_dir, 'model_%d.h5' % i)
        else:
            curr_model.save(os.path.join(config.job_dir, 'model_%d.h5' % i))

        predictions = curr_model.predict(X_train, batch_size=64, verbose=1)
        if config.job_dir.startswith("gs://"):
            np.save('train_predictions_%d.npy' % i, predictions)
            copy_file_to_gcs(config.job_dir, 'train_predictions_%d.npy' % i)
        else:
            np.save(os.path.join(config.job_dir, 'train_predictions_%d.npy' % i), predictions)

        # # Save test predictions
        # predictions = curr_model.predict(X_test, batch_size=64, verbose=1)
        # if config.job_dir.startswith("gs://"):
        #     np.save('test_predictions_%d.npy' % i, predictions)
        #     copy_file_to_gcs(config.job_dir, 'test_predictions_%d.npy' % i)
        # else:
        #     np.save(os.path.join(config.job_dir, 'test_predictions_%d.npy' % i), predictions)

        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        if config.job_dir.startswith("gs://"):
            test[['label']].to_csv('predictions_%d.csv' % i)
            copy_file_to_gcs(config.job_dir, 'predictions_%d.csv' % i)
        else:
            test[['label']].to_csv(os.path.join(config.job_dir, 'predictions_%d.csv' % i))
        # Convert the Keras models to TensorFlow SavedModel
        to_savedmodel(curr_model, os.path.join(config.job_dir, 'export_%d' % i))


def train_network(config, val_split=0.25, epochs=50, batch_size=20):
    np.random.seed(1)

    # Get the data
    X_train, Y_train, paths_train, class_names = build_data_set(config,
                                                                path='gs://cmps140-205021-mlengine/input_preprocessed/audio_train')

    # Instantiate the model
    model = model_fn_vnn(X_train.shape, config)

    checkpoint = ModelCheckpoint(config.job_dir + 'best_1.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True)
    early = EarlyStopping(monitor='val_loss', mode='min', patience=12)

    tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold_1', write_graph=True)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpoint, early, tb],
              validation_split=val_split)  # validation_data=(X_val, Y_val),

    # # Score the model against Test dataset
    # X_test, Y_test, paths_test, class_names_test = build_data_set(config, '../out/audio_test')
    # assert (class_names == class_names_test)
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


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
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    file_stream = file_io.FileIO(config.test_csv[0], mode='r')
    test = pd.read_csv(StringIO(file_stream.read()))
    test['label'] = predicted_labels
    if config.job_dir.startswith("gs://"):
        test[['fname', 'label']].to_csv('1d_2d_ensembled_submission.csv', index=False)
        copy_file_to_gcs(config.job_dir, '1d_2d_ensembled_submission.csv')
    else:
        test[['fname', 'label']].to_csv(os.path.join(config.job_dir, '1d_2d_ensembled_submission.csv'), index=False)


def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2, model_level, n_mfcc,
                  audio_duration):
    config = Config(sampling_rate=44100, audio_duration=audio_duration, n_folds=5,
                    learning_rate=learning_rate, use_mfcc=True, n_mfcc=n_mfcc, train_csv=train_files,
                    test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2,
                    model_level=model_level, validated_labels_only=1)
    train_network(config)
    # create_predictions(config)


# create_config('../input/train.csv',
#               '../input/sample_submission.csv',
#               '../out',
#               0.001,
#               '../input/audio_train/',
#               '../input/audio_test/',
#               1,
#               40,
#               2)

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
                        help='GCS or local dir to write checkpoints and export models')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--model_level',
                        type=int,
                        default=2,
                        help='Which resnet models to use')
    parser.add_argument('--n_mfcc',
                        type=int,
                        default=40,
                        help='nfcc')
    parser.add_argument('--audio_duration',
                        type=int,
                        default=2,
                        help='Audio duration')
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
