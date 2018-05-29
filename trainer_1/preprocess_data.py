'''
Preprocess audio
'''
from __future__ import print_function

import glob

import numpy as np
import librosa
from audioread import NoBackendError
import os
from PIL import Image
from functools import partial
from imageio import imwrite
import multiprocessing as mp
from trainer_2.config import Config


def make_phase_gram(mono_sig, sr, n_bins=128):
    stft = librosa.stft(mono_sig)  # , n_fft = (2*n_bins)-1)
    magnitude, phase = librosa.magphase(stft)  # we don't need magnitude
    # resample the phase array to match n_bins
    phase = np.resize(phase, (n_bins, phase.shape[1]))[np.newaxis, :, :, np.newaxis]
    return phase


def make_melgram(mono_sig, sr, n_mels=128):
    return librosa.amplitude_to_db(librosa.feature.melspectrogram(mono_sig, sr=sr, n_mels=n_mels))[np.newaxis, :, :,
           np.newaxis]


# turn multichannel audio as multiple melgram layers
def make_layered_melgram(signal, sr, mels=128, phase=False):
    if signal.ndim == 1:  # given the way the preprocessing code is  now, this may not get called
        signal = np.reshape(signal, (1, signal.shape[0]))
    # get mel-spectrogram for each channel, and layer them into multi-dim array
    for channel in range(signal.shape[0]):
        melgram = make_melgram(signal[channel], sr, n_mels=mels)
        if 0 == channel:
            layers = melgram
        else:
            layers = np.append(layers, melgram,
                               axis=3)  # we keep axis=0 free for keras batches, axis=3 means 'channels_last'
        if phase:
            phasegram = make_phase_gram(signal[channel], sr, n_bins=mels)
            layers = np.append(layers, phasegram, axis=3)
    return layers


# this is either just the regular shape, or it returns a leading 1 for mono
def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return 1, signal.shape[0]
    else:
        return signal.shape


def find_max_shape(path, mono=False, sr=None, dur=None, clean=False):
    shapes = []
    for dir_name, dir_names, file_names in os.walk(path):
        for filename in file_names:
            if '.wav' in filename:
                file_path = os.path.join(dir_name, filename)
                try:
                    signal, sr = librosa.load(file_path, mono=mono, sr=16000)
                except NoBackendError as e:
                    print("Could not open audio file {}".format(file_path))
                    raise e
                if clean:  # Just take the first file and exit
                    return get_canonical_shape(signal)
                shapes.append(get_canonical_shape(signal))
    return max(s[0] for s in shapes), max(s[1] for s in shapes)


def save_melgram(outfile, melgram, out_format='npy'):
    channels = melgram.shape[1]
    melgram = melgram.astype(np.float16)
    if (('jpeg' == out_format) or ('png' == out_format)) and (channels <= 4):
        melgram = np.moveaxis(melgram, 1, 3).squeeze()
        melgram = np.flip(melgram, 0)
        if 2 == channels:
            b = np.zeros((melgram.shape[0], melgram.shape[1], 3))  # 3-channel array of zeros
            b[:, :, :-1] = melgram  # fill the zeros on the 1st 2 channels
            imwrite(outfile, b, format=out_format)
        else:
            imwrite(outfile, melgram, format=out_format)
    elif 'npy' == out_format:
        np.save(outfile, melgram)
    else:
        np.savez_compressed(outfile, melgram=melgram)  # default is compressed npz file
    return


# class_files, classname, dirname, mono, outpath, max_shape, out_format, mels, phase, file_index
def convert_one_file(config, folder, file_name, max_shape, phase, out_format):
    signal, sr = librosa.load(file_name, sr=config.sampling_rate)
    shape = get_canonical_shape(signal)  # either the signal shape or a leading one
    if shape != signal.shape:  # this only evals to true for mono
        signal = np.reshape(signal, shape)
    padded_signal = np.zeros(max_shape)
    use_shape = list(max_shape[:])
    use_shape[0] = min(shape[0], max_shape[0])
    use_shape[1] = min(shape[1], max_shape[1])
    padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]
    layers = make_layered_melgram(padded_signal, sr, mels=config.mels, phase=phase)

    outfile = config.job_dir + '/' + folder + '/' + file_name.rpartition('/')[2].replace('.wav', '') + '.' + out_format
    save_melgram(outfile, layers, out_format=out_format)
    return


def preprocess_dataset(config, phase=False, out_format='npy'):
    print("here")
    max_shape = min(find_max_shape(config.train_dir, mono=True),
                    find_max_shape(config.test_dir, mono=True))
    print(max_shape)
    sampleset_subdirs = ['audio_train/', 'audio_test/']
    train_outpath = config.job_dir + "/audio_train/"
    test_outpath = config.job_dir + "/audio_test/"
    if not os.path.exists(config.job_dir):
        os.mkdir(config.job_dir)
        os.mkdir(train_outpath)
        os.mkdir(test_outpath)
    for sub_dir in sampleset_subdirs:
        files = glob.glob(os.path.join('../input/', sub_dir, '*.wav'))
        for i, file_name in enumerate(files):
            convert_one_file(config, sub_dir.replace('/', ''), file_name, max_shape, phase, out_format)
    return


def create_config(train_files, eval_files, job_dir, learning_rate, user_arg_1, user_arg_2, model_level, n_mfcc,
                  audio_duration):
    config = Config(sampling_rate=44100, audio_duration=audio_duration, n_folds=10,
                    learning_rate=learning_rate, use_mfcc=True, n_mfcc=n_mfcc, train_csv=train_files,
                    test_csv=eval_files, job_dir=job_dir, train_dir=user_arg_1, test_dir=user_arg_2,
                    model_level=model_level, validated_labels_only=1)
    preprocess_dataset(config)

create_config('../input/train.csv', '../input/sample_submission.csv', './out', 0.001, '../input/audio_train/',
              '../input/audio_test/', 1, 40, 2)
