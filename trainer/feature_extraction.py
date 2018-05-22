import os

import librosa
import numpy as np
import soundfile as sf
from tensorflow.python.lib.io import file_io

from trainer.task import copy_file_to_gcs


def extract_feature(data, sample_rate):
    # short term fourier transform
    stft = np.abs(librosa.stft(data))
    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def prepare_data(df, config, data_dir, save_load=False):
    if config.job_dir.startswith('gs://') and is_file_available(data_dir + 'input_arr.npy') and save_load:
        with file_io.FileIO(data_dir + 'input_arr.npy', mode='r') as input_f:
            return np.load(input_f)
    elif not config.job_dir.startswith('gs://') and os.path.exists(
            os.path.join(data_dir, 'input_arr.npy')) and save_load:
        return np.load(os.path.join(data_dir, 'input_arr.npy'))
    else:
        x = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
        input_length = config.audio_length
        for i, fname in enumerate(df.index):
            file_path = data_dir + fname
            if config.job_dir.startswith('gs://'):
                with file_io.FileIO(file_path, mode='r') as input_f:
                    data, samplerate = sf.read(input_f)
                    data = data.T
                    data = librosa.resample(data, samplerate, config.sampling_rate)
            else:
                data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
                # Random offset / Padding
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length + offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                    data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

                data = librosa.feature.melspectrogram(data, sr=config.sampling_rate)
                data = np.expand_dims(data, axis=-1)
                x[i,] = data
            if config.job_dir.startswith('gs://') and save_load:
                np.save('input_arr.npy', x)
                copy_file_to_gcs(data_dir, 'input_arr.npy')
            elif save_load:
                np.save(os.path.join(data_dir, 'input_arr.npy'), x)
        return x


def normalize_data(input_dat):
    mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std
