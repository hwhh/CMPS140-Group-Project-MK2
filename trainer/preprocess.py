import librosa
import soundfile as sf

from trainer.utils import *


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(df, config, data_dir, save_dir, test, bands=128, frames=41):
    log_specgrams, l1 = load_file(data_dir, '/norm_features.npy')
    labels, l2 = load_file(data_dir, '/labels.npy')
    f, l3 = load_file(data_dir, '/f.npy')
    if l1 and l2 and l3:
        return log_specgrams, labels, f
    else:
        log_specgrams, labels, f = [], [], []
        window_size = 512 * (frames - 1)
        for i, fname in enumerate(df.index):
            log_specgrams = []
            file_path = data_dir + fname

            sound_clip, _ = librosa.core.load(file_path, sr=config.sampling_rate)

            if config.job_dir.startswith('gs://'):
                with file_io.FileIO(file_path, mode='r') as input_f:
                    data, samplerate = sf.read(input_f)
                    sound_clip = data.T
                    sound_clip = librosa.resample(sound_clip, samplerate, config.sampling_rate)
            else:
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
                if not test:
                    label.append(df.label_idx[fname])
                    f.append(df.label_idx[fname])
            if not test:
                labels.append(label)
            save_file(save_dir, log_specgrams, '/' + fname.replace('.wav', '.npy'))

        # log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
        # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
        # for i in range(len(features)):
        #     features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        # norm_features = normalize_data(np.array(features))
        # save_file(data_dir, norm_features, '/norm_features.npy')
        # if not test:
        #     save_file(data_dir, np.array(labels), '/labels.npy')
        #     save_file(data_dir, np.array(f), '/f.npy')
        # return norm_features, np.array(labels), np.array(f)
        return None, None, None


def load_arrys(df, data_dir, test):
    log_specgrams, labels, f = [], [], []
    for i, fname in enumerate(df.index):
        file_path = data_dir + fname.replace('.wav', '.npy')
        label = []
        for arr in np.load(file_path):
            log_specgrams.append(arr)
            if not test:
                label.append(df.label_idx[fname])
                f.append(df.label_idx[fname])
        if not test:
            labels.append(label)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    norm_features = normalize_data(np.array(features))
    save_file(data_dir, norm_features, '/norm_features.npy')
    if not test:
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
