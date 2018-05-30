from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, train_test_split

from trainer.batch_generator import DataGenerator
from trainer.config import Config
from trainer.model_1 import model_fn_vnn
from trainer.preprocess import extract_features, one_hot_encode
from trainer.utils import *


def run(config):
    train = pd.read_csv("../input/train_1.csv")
    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index('fname', inplace=True)
    train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])
    ############################################################################
    X_train, X_test, y_train, y_test = train_test_split(np.array(train.label_idx.keys()), train.label_idx,
                                                        test_size=0.2)
    # Datasets
    partition = {'train': X_train, 'test': X_test}
    # Generators
    training_generator = DataGenerator(config, partition['train'], train.label_idx)
    validation_generator = DataGenerator(config, partition['test'], train.label_idx)

    model = model_fn_vnn(config)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=1)


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


create_config('../input/train.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train_1/',
              '../input/audio_test/', 1, 40, 2)