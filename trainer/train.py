from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import StratifiedKFold

from trainer.config import Config
from trainer.model_1 import model_fn_vnn
from trainer.preprocess import extract_features, one_hot_encode
from trainer.utils import *
from trainer.utils import to_savedmodel

os.chdir('/Users/henryhargreaves/Documents/University/Year_3/CMPS140/Group_Project/Models/trainer')

def run(config):
    if config.job_dir.startswith('gs://'):
        file_stream = file_io.FileIO(config.test_csv[0], mode='r')
        test = pd.read_csv(StringIO(file_stream.read()))
        file_stream = file_io.FileIO(config.train_csv[0], mode='r')
        train = pd.read_csv(StringIO(file_stream.read()))
    else:
        test = pd.read_csv("../input/sample_submission.csv")
        train = pd.read_csv("../input/train.csv")

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}

    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

    # x_train, tr_labels, frames = extract_features(train, config, config.train_dir, '../input/audio_train_preprocessed', False)
    # x_test, _, _ = extract_features(test, config, config.test_dir, '../input/audio_test_preprocessed', True)

    load_arrys(train, '../input/audio_train_preprocessed', , False)
    load_arrys(test, '../input/audio_test_preprocessed', , True)

    tr_labels = one_hot_encode(tr_labels, frames, np.unique(train.label_idx))

    skf = StratifiedKFold(n_splits=2).split(np.zeros(len(x_train)), frames)
    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        x, y, x_val, y_val = x_train[train_split], tr_labels[train_split], np.array(x_train[val_split]), tr_labels[
            val_split]

        checkpoint = ModelCheckpoint(config.job_dir + '/best_%d.h5' % i, monitor='val_loss', verbose=1,
                                     save_best_only=True)
        early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        tb = TensorBoard(log_dir=np.os.path.join(config.job_dir, 'logs') + '/fold_%i' % i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]

        print(('Fold: %d' % i) + '\n' + '#' * 50)

        curr_model = model_fn_vnn(x_train.shape, config)

        curr_model.fit(x, y, validation_data=(x_val, y_val), callbacks=callbacks_list,
                       batch_size=64, epochs=config.max_epochs)

        curr_model.load_weights(config.job_dir + '/best_%d.h5' % i)
        if config.job_dir.startswith('gs://'):
            curr_model.save('model_%d.h5' % i)
            copy_file_to_gcs(config.job_dir, 'model_%d.h5' % i)
        else:
            curr_model.save(os.path.join(config.job_dir, 'model_%d.h5' % i))

        predictions = curr_model.predict(x_train, batch_size=64, verbose=1)
        if config.job_dir.startswith("gs://"):
            np.save('train_predictions_%d.npy' % i, predictions)
            copy_file_to_gcs(config.job_dir, 'train_predictions_%d.npy' % i)
        else:
            np.save(os.path.join(config.job_dir, 'train_predictions_%d.npy' % i), predictions)

        # Save test predictions
        predictions = curr_model.predict(x_test, batch_size=64, verbose=1)
        if config.job_dir.startswith("gs://"):
            np.save('test_predictions_%d.npy' % i, predictions)
            copy_file_to_gcs(config.job_dir, 'test_predictions_%d.npy' % i)
        else:
            np.save(os.path.join(config.job_dir, 'test_predictions_%d.npy' % i), predictions)

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


def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2, model_level, n_mfcc,
                  audio_duration):
    config = Config(sampling_rate=44100, audio_duration=audio_duration, n_folds=10,
                    learning_rate=learning_rate, use_mfcc=True, n_mfcc=n_mfcc, train_csv=train_files,
                    test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2,
                    model_level=model_level, validated_labels_only=1)

    try:
        os.makedirs(config.job_dir)
    except OSError:
        pass

    run(config)


create_config('../input/train.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train/',
              '../input/audio_test/', 1, 40, 2)
