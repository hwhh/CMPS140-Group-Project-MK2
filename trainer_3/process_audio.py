import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.core

"""
Window (FFT) sizes used are 23ms (1024), 11.5ms (512), 46ms (2048), and 92ms
These will make up 4 channels, currently just using 2048

Change chanel to 2 to take into account librosa.feature.delta

could flattern the array ?
"""


def prepare_data(df, config, data_dir, bands=128):
    x = np.empty(shape=(len(df.index), bands))
    log_specgrams_2048 = []
    for i, fname in enumerate(df.index):
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=16000, res_type="kaiser_fast")
        melspec = librosa.feature.melspectrogram(data, sr=16000, n_mels=bands)
        logspec = librosa.core.power_to_db(melspec)
        log_specgrams_2048.append(logspec)
    return log_specgrams_2048


def normalize_data(input_dat):
    mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std


test = pd.read_csv("../input/sample_submission.csv")
train = pd.read_csv("../input/train_1.csv")

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}

train.set_index('fname', inplace=True)
test.set_index('fname', inplace=True)
train['label_idx'] = train.label.apply(lambda elem: label_idx[elem])

x_train = prepare_data(train, None, '../input/audio_train_1/')
