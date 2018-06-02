import os

import keras
import librosa
import numpy as np
import soundfile as sf
from tensorflow.python.lib.io import file_io


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, config, list_IDs, labels, data_dir, type, batch_size=128, dim=(128, 41), n_channels=2,
                 shuffle=True):
        """Initialization"""
        self.dim = dim
        self.type = type
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = config.n_classes
        self.config = config
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.on_epoch_end()

    @staticmethod
    def normalize_data(input_dat):
        # Extracting single channels from 2 channel file
        channel_1 = input_dat[:, :, :, 0]
        channel_2 = input_dat[:, :, :, 1]
        # normalizing per channel data:
        channel_1 = (channel_1 - np.min(channel_1)) / (np.max(channel_1) - np.min(channel_1))
        channel_2 = (channel_2 - np.min(channel_2)) / (np.max(channel_2) - np.min(channel_2))
        # putting the 2 channels back together:
        audio_norm = np.empty((input_dat.shape[0], input_dat.shape[1], input_dat.shape[2], 2), dtype=np.float32)
        audio_norm[:, :, :, 0] = channel_1
        audio_norm[:, :, :, 1] = channel_2
        return np.nan_to_num(audio_norm)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.type == 0:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Generate data
        X, labels = self.extract_features(list_IDs_temp, self.data_dir)
        return np.array(X), keras.utils.to_categorical(labels, num_classes=self.config.n_classes)  # TODO

    def extract_features(self, df, data_dir, sr=44100, bands=128):
        X, labels = [], []
        # Convert once
        for fname in df:
            file_path = os.path.join(data_dir, fname)
            if self.config.job_dir.startswith('gs://'):
                with file_io.FileIO(file_path, mode='r') as input_f:
                    data, samplerate = sf.read(input_f)
                    data = data.T
                    data = librosa.resample(data, sr, sr)
            else:
                data, _ = librosa.core.load(file_path, sr=sr, res_type="kaiser_fast")

            melspec = librosa.feature.melspectrogram(data, sr=sr, n_mels=bands)
            logspec = librosa.core.power_to_db(melspec)  # shape would be [128, your_audio_length]
            logspec = logspec[..., np.newaxis]  # shape will be [128, your_audio_length, 1]
            X.append(logspec)
            labels.append(self.labels[fname])
        # Find longest
        max_length = np.max([x.shape[1] for x in X])
        # Pad zero to make them all the same length
        X2 = [np.pad(x, ((0, 0), (0, max_length - x.shape[1]), (0, 0)), 'constant') for x in X]
        features = np.concatenate((X2, np.zeros(np.shape(X2))), axis=3)
        for i in range(len(X2)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        features = self.normalize_data(features)
        return np.array(features), labels
