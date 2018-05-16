import json
import sys
import librosa
import numpy as np
from keras.models import load_model

LABELS = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock', 'Gunshot_or_gunfire', 'Clarinet',
          'Computer_keyboard', 'Keys_jangling', 'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute',
          'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak', 'Scissors', 'Harmonica', 'Gong',
          'Microwave_oven', 'Burping_or_eructation', 'Double_bass', 'Shatter', 'Fireworks', 'Tambourine', 'Cowbell',
          'Electric_piano', 'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar', 'Violin_or_fiddle',
          'Finger_snapping']


def normalize_data(input_dat):
    mean = np.mean(input_dat, axis=0)
    std = np.std(input_dat, axis=0)
    return (input_dat - mean) / std


def process_audio_file():
    input_length = 88200
    data, _ = librosa.core.load(sys.argv[1], sr=44100, res_type="kaiser_fast")
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

    data = librosa.feature.mfcc(data, sr=16000, n_mfcc=40)
    data = np.expand_dims(data, axis=-1)

    data = normalize_data(data)
    data = data[np.newaxis, ...]
    return data


def predict():
    X = process_audio_file()
    model = load_model('../model/mfcc_resdiual_21-model_2.h5')
    predictions = model.predict(X, batch_size=64, verbose=1)
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    print(predicted_labels)


predict()