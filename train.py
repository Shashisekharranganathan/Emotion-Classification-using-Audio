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


# Loading all the File Paths
angry = glob.glob("Emotions Dataset/Angry/*.wav")
disgusted = glob.glob("Emotions Dataset/Disgusted/*.wav")
fearful = glob.glob("Emotions Dataset/Fearful/*.wav")
happy = glob.glob("Emotions Dataset/Happy/*.wav")
neutral = glob.glob("Emotions Dataset/Neutral/*.wav")
sad = glob.glob("Emotions Dataset/Sad/*.wav")
suprised = glob.glob("Emotions Dataset/Suprised/*.wav")


# Removing Corrupted WAV Files
angry_files = []
happy_files = []
sad_files = []
neutral_files = []
disgusted_files = []
suprised_files = []
fearful_files = []

for i in range(len(angry)):
    try:
        file_contents = tf.io.read_file(angry[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        angry_files.append(angry[i])
    except:
        pass

for i in range(len(happy)):
    try:
        file_contents = tf.io.read_file(happy[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        happy_files.append(happy[i])
    except:
        pass

for i in range(len(neutral)):
    try:
        file_contents = tf.io.read_file(neutral[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        neutral_files.append(neutral[i])
    except:
        pass

for i in range(len(sad)):
    try:
        file_contents = tf.io.read_file(sad[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        sad_files.append(sad[i])
    except:
        pass

for i in range(len(suprised)):
    try:
        file_contents = tf.io.read_file(suprised[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        suprised_files.append(suprised[i])
    except:
        pass

for i in range(len(disgusted)):
    try:
        file_contents = tf.io.read_file(disgusted[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        disgusted_files.append(disgusted[i])
    except:
        pass

for i in range(len(fearful)):
    try:
        file_contents = tf.io.read_file(fearful[i])
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=1)
        fearful_files.append(fearful[i])
    except:
        pass

print(f"Sad: {len(sad_files)}")
print(f"Happy: {len(happy_files)}")
print(f"Angry: {len(angry_files)}")
print(f"Fearful: {len(fearful_files)}")
print(f"Suprised: {len(happy_files)}")
print(f"Neutral: {len(neutral_files)}")
print(f"Disgusted: {len(disgusted_files)}")

# Loading Paths to make Tensorflow Dataset
SAD = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(sad_files))
HAPPY = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(happy_files))
ANGRY = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(angry_files))
NEUTRAL = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(neutral_files))
DISGUSTED = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(disgusted_files))
SUPRISED = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(suprised_files))
FEARFUL = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(fearful_files))

# Labels aka Targets
labels = ["SAD", "HAPPY", "ANGRY", "NEUTRAL",
          "DISGUSTED", "SUPRISED", "FEARFUL"]
ys_labels = [0, 1, 2, 3, 4, 5, 6]
ys_labels = tf.keras.utils.to_categorical(ys_labels)

# TF Dataset Loaders
sad_loader = tf.data.Dataset.zip((SAD, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[0] for _ in range(len(SAD))]))))
happy_loader = tf.data.Dataset.zip((HAPPY, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[1] for _ in range(len(HAPPY))]))))
angry_loader = tf.data.Dataset.zip((ANGRY, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[2] for _ in range(len(ANGRY))]))))
neutral_loader = tf.data.Dataset.zip((NEUTRAL, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[3] for _ in range(len(NEUTRAL))]))))
disgusted_loader = tf.data.Dataset.zip((DISGUSTED, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[4] for _ in range(len(DISGUSTED))]))))
suprised_loader = tf.data.Dataset.zip((SUPRISED, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[5] for _ in range(len(SUPRISED))]))))
fearful_loader = tf.data.Dataset.zip((FEARFUL, tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor([ys_labels[6] for _ in range(len(FEARFUL))]))))

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


# Dataset Pipeling for training
dataset = angry_loader.concatenate(sad_loader)
dataset = dataset.concatenate(happy_loader)
dataset = dataset.concatenate(neutral_loader)
dataset = dataset.concatenate(disgusted_loader)
dataset = dataset.concatenate(suprised_loader)
dataset = dataset.concatenate(fearful_loader)
print(f"Total Records: {len(dataset)}")
dataset = dataset.map(preprocess)
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(8)
dataset = dataset.prefetch(2)

train = dataset.take(100)
test = dataset.skip(100).take(10)


# Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    16, (3, 3), activation="relu", input_shape=(1522, 257, 1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(len(ys_labels), activation="softmax"))

# Compiling
model.compile(
    optimizer="Adam",
    loss="CategoricalCrossentropy",
    metrics=[
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()
    ]
)

# Callbacks
earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = tf.keras.callbacks.ModelCheckpoint(
    'model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

history = model.fit(train, epochs=5, validation_data=test, callbacks=[
                    earlyStopping, mcp_save, reduce_lr_loss])
