import argparse
import os

from trainer.config import Config
from trainer.train import run, run_with_k_fold


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

    run_with_k_fold(config)


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
