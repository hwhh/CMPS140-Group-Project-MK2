import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# y, sr = librosa.load('./input/audio_train/1e1c1a81.wav', duration=10)
# plt.figure()
# plt.subplot(3, 1, 1)
# librosa.display.waveplot(y, sr=sr)
# plt.title('Gun Shots')
# plt.show()
#
# y, sr = librosa.load('./input/audio_train/16833489.wav', duration=10)
# melspec = librosa.feature.melspectrogram(y)
# logspec = librosa.core.power_to_db(melspec)  # shape would be [128, your_audio_length]
# plt.figure()
# plt.subplot(3, 1, 1)
# librosa.display.specshow(melspec, sr=sr)
# plt.title('Firework')
# plt.show()
#



y, sr = librosa.load('./input/audio_train/16833489.wav')
plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()