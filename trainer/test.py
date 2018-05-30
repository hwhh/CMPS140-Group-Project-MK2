import multiprocessing
import warnings

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.model_selection import train_test_split

from trainer.batch_generator import DataGenerator
from trainer.config import Config
from trainer.model_1 import model_fn_vnn
from trainer.utils import *

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def run(config):
    if config.job_dir.startswith('gs://'):
        file_stream = file_io.FileIO(config.test_csv[0], mode='r')
        test = pd.read_csv(StringIO(file_stream.read()))
        file_stream = file_io.FileIO(config.train_csv[0], mode='r')
        train = pd.read_csv(StringIO(file_stream.read()))
    else:
        test = pd.read_csv(config.test_csv)
        train = pd.read_csv(config.train_csv)

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index('fname', inplace=True)
    test.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])
    X_train, X_test, y_train, y_test = train_test_split(np.array(train.label_idx.keys()), train.label_idx,
                                                        test_size=0.2)
    partition = {'train': X_train, 'test': X_test}
    training_generator = DataGenerator(config, partition['train'], train.label_idx, config.train_dir, 0)
    test_generator = DataGenerator(config, partition['test'], train.label_idx, config.train_dir, 0)

    model = model_fn_vnn(config)

    checkpoint = ModelCheckpoint(config.job_dir + '/best.h5', monitor='val_loss', verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir=os.path.join(config.job_dir, 'logs') + '/fold', write_graph=True)
    callbacks_list = [checkpoint, tb]

    model.fit_generator(generator=training_generator,
                        validation_data=test_generator,
                        use_multiprocessing=True,
                        workers=1,
                        epochs=50,
                        callbacks=callbacks_list)

    if config.job_dir.startswith('gs://'):
        model.save('model.h5')
        copy_file_to_gcs(config.job_dir, 'model.h5')
    else:
        model.save(os.path.join(config.job_dir, 'model.h5'))

    create_predictions(training_generator, model, config, test, LABELS)


def create_predictions(training_generator, model, config, test, labels):
    predictions = model.predict_generator(generator=training_generator)
    if config.job_dir.startswith("gs://"):
        np.save('train_predictions.npy', predictions)
        copy_file_to_gcs(config.job_dir, 'train_predictions.npy')
    else:
        np.save(os.path.join(config.job_dir, 'train_predictions.npy'), predictions)

    test_generator = DataGenerator(config, test.index, None, config.test_dir, 1)
    predictions = model.predict_generator(generator=test_generator)
    if config.job_dir.startswith("gs://"):
        np.save('test_predictions.npy', predictions)
        copy_file_to_gcs(config.job_dir, 'test_predictions.npy')
    else:
        np.save(os.path.join(config.job_dir, 'test_predictions.npy'), predictions)
    create_submission(model, config, test, labels, predictions)


def create_submission(model, config, test, labels, predictions):
    top_3 = np.array(labels)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    if config.job_dir.startswith("gs://"):
        test[['label']].to_csv('predictions.csv')
        copy_file_to_gcs(config.job_dir, 'predictions.csv')
    else:
        test[['label']].to_csv(os.path.join(config.job_dir, 'predictions.csv'))
    to_savedmodel(model, os.path.join(config.job_dir, 'export'))


def create_confusion_matrix():
    pass
    # y_true = np.array([0] * 1000 + [1] * 1000)
    # y_pred = probabilities > 0.5
    # confusion_matrix(y_true, y_pred)


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


create_config('../input/train_1.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train_1/',
              '../input/audio_test/', 1, 40, 2)
