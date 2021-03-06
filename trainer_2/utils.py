import glob
import os

import librosa
import numpy as np
import soundfile as sf
from keras import backend as K
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io.file_io import stat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 models into TensorFlow SavedModel."""
    builder = saved_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def is_file_available(filepath):
    try:
        return stat(filepath)
    except NotFoundError as e:
        return False


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


def prepare_data_fixed(df, config, data_dir, save_load=False):
    if config.job_dir.startswith('gs://') and is_file_available(data_dir + 'input_arr.npy') and save_load:
        with file_io.FileIO(data_dir + 'input_arr.npy', mode='r') as input_f:
            return np.load(input_f)
    elif not config.job_dir.startswith('gs://') and os.path.exists(
            os.path.join(data_dir, 'input_arr.npy')) and save_load:
        return np.load(os.path.join(data_dir, 'input_arr.npy'))
    else:
        x = np.empty(shape=(df.shape[0], 128, 173, 1))
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


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def prepare_data_window(parent_dir, sub_dirs, labels, file_ext="*.wav", bands=20, frames=41):
    # if not parent_dir.startswith('gs://') and os.path.exists(parent_dir + sub_dirs[0] + '/features.npy'):
    #     return np.load(parent_dir + sub_dirs[0] + '/features.npy'), np.load(parent_dir + sub_dirs[0] + '/labels.npy'),
    window_size = 512 * (frames - 1)
    mfccs, window_labels = [], []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            if labels is not None:
                label = labels[fn.rpartition('/')[2]]
            for (start, end) in windows(sound_clip, window_size):
                if len(sound_clip[int(start):int(end)]) == window_size:
                    signal = sound_clip[int(start):int(end)]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc=bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    if labels is not None:
                        window_labels.append(label)
    features = np.asarray(mfccs).reshape(len(mfccs), frames, bands)
    save_files(parent_dir, sub_dirs, features, labels)
    return np.array(features), np.array(window_labels, dtype=np.int)


def save_files(parent_dir, sub_dirs, features, labels):
    if parent_dir.startswith('gs://'):
        np.save('features.npy', np.array(features))
        copy_file_to_gcs(parent_dir + sub_dirs[0], 'features.npy')
        if labels is not None:
            np.save('labels.npy', np.array(labels))
            copy_file_to_gcs(parent_dir + sub_dirs[0], 'labels.npy')
    else:
        np.save(parent_dir + sub_dirs[0] + '/features.npy', np.array(features))
        if labels is not None:
            np.save(parent_dir + sub_dirs[0] + '/labels.npy', np.array(labels))


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def normalize_data(input_dat):
    mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std


def load_data(df, config, data_dir):
    x = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], config.dim[2]))
    for i, fname in enumerate(df.index):
        file_path = data_dir + fname.replace('.wav', '.npy')
        if config.job_dir.startswith('gs://'):
            with file_io.FileIO(file_path, mode='r') as input_f:
                data = np.load(input_f)
        else:
            data = np.load(file_path)
        x[i,] = data
    return x
