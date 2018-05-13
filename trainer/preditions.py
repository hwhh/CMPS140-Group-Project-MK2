import json
import sys
import librosa
import numpy as np
from keras.models import load_model

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    # data = data[np.newaxis, ...]
    return data


# X = process_audio_file()
# model = load_model('../mfcc_resdiual_21-model_2.h5')
# yhat = model.predict(X, verbose=0)
# print(yhat)

# print(json.dumps({"input": process_audio_file().tolist()}))

x = [0.0038368015084415674, 0.00498126121237874, 0.34241732954978943, 0.0004900456988252699, 0.0028206747956573963,
     0.03744399547576904, 0.013452275656163692, 3.0280185455922037e-05, 0.0010292797815054655, 0.0017782801296561956,
     0.02276207134127617, 0.0034001385793089867, 0.0015485514886677265, 0.0004882702196482569, 0.015835274010896683,
     0.0026223016902804375, 0.001943460083566606, 0.0005182517925277352, 0.006459018215537071, 0.00394971389323473,
     0.0008023412665352225, 0.009913447313010693, 0.00039583357283845544, 0.045332953333854675, 0.0015668823616579175,
     0.00013052560098003596, 0.327932745218277, 0.0008870949968695641, 0.0008640772430226207, 0.029977643862366676,
     0.006315059959888458, 0.056280605494976044, 0.013677951879799366, 0.0010627773590385914, 0.007754478137940168,
     0.00042011638288386166, 0.0022167996503412724, 0.005968049168586731, 0.007246673107147217, 3.865496182697825e-05,
     0.01340796984732151]
big = -1
pos = -1
for (count, val) in enumerate(x):
    if val > big:
        big = val
        pos = count

print(big)
print(pos)
