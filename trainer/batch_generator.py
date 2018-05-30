import librosa
import numpy as np
import keras

from trainer.preprocess import windows, one_hot_encode


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, config, list_IDs, labels, batch_size=2, dim=(128, 41), n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = config.n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X, labels, frames = self.extract_features(list_IDs_temp, '../input/audio_train/')
        return X, one_hot_encode(labels, frames, np.unique(self.labels))

    def extract_features(self, df, data_dir, bands=128, frames=41, sr=22050):
        log_specgrams, labels, f = [], [], []
        window_size = 512 * (frames - 1)
        for fname in df:
            file_path = data_dir + fname
            sound_clip, _ = librosa.core.load(file_path)
            label = []
            for (start, end) in (windows(sound_clip, window_size)):
                if len(sound_clip[int(start):int(end)]) == window_size:
                    signal = sound_clip[int(start):int(end)]
                elif len(sound_clip[int(start):int(end)]) < window_size:
                    max_offset = window_size - len(sound_clip[int(start):int(end)])
                    offset = np.random.randint(max_offset)
                    signal = np.pad(sound_clip[int(start):int(end)],
                                    (offset, window_size - len(sound_clip[int(start):int(end)]) - offset), 'constant',
                                    constant_values=0)
                melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                logspec = librosa.power_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                label.append(self.labels[fname])
                f.append(self.labels[fname])
            labels.append(label)

        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        return np.array(features), np.array(labels), f
