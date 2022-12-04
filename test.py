import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile

# Loading the Audio WAV Files


def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# Function that generates the spectrograms of the WAV
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:49000]
    zero_padding = tf.zeros([49000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram, label

model = tf.keras.models.load_model("model.hdf5")
spectrogram, label = preprocess("", label=True)

y_pred = model.predict(spectrogram)
labels = ["SAD", "HAPPY", "ANGRY", "NEUTRAL", "DISGUSTED", "SUPRISED", "FEARFUL"]
print(f"The prediction is : {labels[np.argmax(y_pred)]}")


