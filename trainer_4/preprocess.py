import os

import librosa
import numpy as np
from tensorflow.python.lib.io import file_io


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(df, config, data_dir, bands=128, frames=41):
    if config.job_dir.startswith('gs://') and is_file_available(data_dir + 'input_arr.npy') and save_load:
        with file_io.FileIO(data_dir + 'input_arr.npy', mode='r') as input_f:
            return np.load(input_f)
    elif not config.job_dir.startswith('gs://') and os.path.exists(
            os.path.join(data_dir, 'input_arr.npy')) and save_load:
        return np.load(os.path.join(data_dir, 'input_arr.npy'))
    else:
        window_size = 512 * (frames - 1)
        log_specgrams, labels, f = [], [], []
        for i, fname in enumerate(df.index):
            file_path = data_dir + fname
            sound_clip, _ = librosa.core.load(file_path, sr=config.sampling_rate)
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
                label.append(df.label_idx[fname])
                f.append(df.label_idx[fname])
            labels.append(label)
        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        norm_features = normalize_data(np.array(features))
        save_file(data_dir, norm_features, '/features.npy')
        save_file(data_dir, np.array(labels), '/labels.npy')
        save_file(data_dir, np.array(f), '/f.npy')
        return norm_features, np.array(labels), np.array(f)


def one_hot_encode(labels, frames, unique_labels):
    n_labels = len(frames)
    n_unique_labels = len(unique_labels)
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    start = 0
    for l in labels:
        one_hot_encode[start:start + len(l), l[0]:l[0] + 1] = 1
        start += len(l)
    return one_hot_encode


def normalize_data(input_dat):
    mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std
