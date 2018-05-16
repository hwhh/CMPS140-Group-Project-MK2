import numpy as np


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20, train_csv='', test_csv='', job_dir='',
                 train_dir='', test_dir='', model_level=2, validated_labels_only=0):

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.job_dir = job_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.validated_labels_only = validated_labels_only
        self.model_level = model_level

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length / 512)), 1)
        else:
            self.dim = (self.audio_length, 1)
