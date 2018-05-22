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
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""
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
    # X, sample_rate = sf.read(file_name, dtype='float32')
    # if X.ndim > 1:
    #     X = X[:, 0]
    # X = X.T

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


def prepare_data(df, config, data_dir):
    # if config.job_dir.startswith('gs://') and is_file_available(data_dir + 'input_arr.npy'):
    #     with file_io.FileIO(data_dir + 'input_arr.npy', mode='r') as input_f:
    #         return np.load(input_f)
    # elif not config.job_dir.startswith('gs://') and os.path.exists(os.path.join(data_dir, 'input_arr.npy')):
    #     return np.load(os.path.join(data_dir, 'input_arr.npy'))
    # else:
    x = np.empty(shape=(128, 193, 1))
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

            data1 = librosa.feature.mfcc(data, sr=config.sampling_rate)
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(data, config.sampling_rate)
            data = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            data = np.expand_dims(data, axis=-1)
            x[i,] = data
        # if config.job_dir.startswith('gs://'):
        #     np.save('input_arr.npy', x)
        #     copy_file_to_gcs(data_dir, 'input_arr.npy')
        # else:
        #     np.save(os.path.join(data_dir, 'input_arr.npy'), x)
        return x


def normalize_data(input_dat):
    # mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std
