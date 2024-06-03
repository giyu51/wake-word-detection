import argparse
import os
import pyaudio
import wave

import numpy as np
import tensorflow as tf
import librosa

import argparse
import os
import pyaudio
import wave
import time


def spectrogramFromAudioData(
    audio_data: np.ndarray,
    sr: int = 44100,
    expand_last_dim=False,
    pre_emphasis_coef: None | float = None,
    use_normalization: bool = True,
):
    # Load the audio data
    if pre_emphasis_coef is not None:
        audio_data = librosa.effects.preemphasis(audio_data, coef=pre_emphasis_coef)

    # Spectrogram generation using Mel-frequency cepstral coefficients (MFCCs)
    spectrogram = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)

    if use_normalization:
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min()
        )

    if expand_last_dim:
        spectrogram = np.expand_dims(spectrogram, -1)

    return spectrogram


def spectrogramFromFile(
    audio_filepath: str,
    sr: int = 44100,
    expand_last_dim=False,
    pre_emphasis_coef: None | float = None,
    use_normalization: bool = True,
):
    # Load the audio data
    audio, sr = librosa.load(audio_filepath, sr=sr)

    if pre_emphasis_coef is not None:
        audio = librosa.effects.preemphasis(audio, coef=pre_emphasis_coef)

    # Spectrogram generation using Mel-frequency cepstral coefficients (MFCCs)
    spectrogram = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if use_normalization:
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min()
        )

    if expand_last_dim:
        spectrogram = np.expand_dims(spectrogram, -1)

    return spectrogram


def _predict(sample_batch):

    prediction = model.predict(sample_batch, verbose=0)
    prediction = prediction[-1][0]
    prediction = np.round(prediction, 2)
    return prediction


def process_audio(audio_file):
    """
    Process the audio file using the specified model.

    Args:
        audio_file (str): Path to the audio file.
        model_path (str): Path to the model file.
        batch_size (int): Batch size for processing (default: 32).
        sample_shape (tuple): Shape of audio samples (default: (40, 173)).
    """
    # Add processing code here
    audio_data = spectrogramFromFile(audio_file, pre_emphasis_coef=0.95)
    batch_placeholder: np.ndarray = np.zeros(
        shape=(args.batch_size - 1,) + sample_shape
    )

    data_batch = np.vstack((batch_placeholder, audio_data[np.newaxis, :, :]))

    return _predict(data_batch)


def record_audio(model_path, batch_size=32, sample_shape=(40, 173)):
    """
    Record audio using the microphone and process it using the specified model.

    Args:
        model_path (str): Path to the model file.
        batch_size (int): Batch size for processing (default: 32).
        sample_shape (tuple): Shape of audio samples (default: (40, 173)).
    """
    # Add recording and processing code here
    pass


def is_wav_file(filename):
    """
    Check if the given file is in WAV format.

    Args:
        filename (str): Path to the file.

    Returns:
        bool: True if the file is in WAV format, False otherwise.
    """
    return filename.lower().endswith(".wav")


def record_audio():
    """
    Record audio using the microphone.
    """
    # Record audio for 2 seconds
    duration = 2  # seconds
    sample_rate = 44100
    chunk = 1024
    format = pyaudio.paInt16

    audio = pyaudio.PyAudio()

    try:
        stream = audio.open(
            format=format,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
        )

        print("Get ready, recording in 1 second...")
        time.sleep(1)

        print("🟢 Recording...")

        frames = []
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        print("🔴 Finished recording.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Save recorded audio to a WAV file
    output_file = "recorded_audio.wav"
    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    return output_file


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process audio with a specified model.")
parser.add_argument(
    "--audio", nargs="?", default=None, help="Path to the audio file (default: None)"
)
parser.add_argument(
    "--model_path",
    nargs="?",
    default="model.keras",
    help="Path to the model file (default: model.keras)",
)
parser.add_argument(
    "--use_recorder",
    action="store_true",
    help="Use voice recorder instead of audio file (default: False)",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for processing (default: 32)"
)
parser.add_argument(
    "--sample_shape",
    type=str,
    default="40,173",
    help="Shape of audio samples (default: (40,173))",
)
args = parser.parse_args()

# Check if either audio file or use_recorder is specified
if args.audio is None and not args.use_recorder:
    parser.error("You must specify either 'audio' file or 'use_recorder' option")

if args.audio is not None and not is_wav_file(args.audio):
    parser.error("The audio file must be in WAV format")

sample_shape = tuple(map(int, args.sample_shape.split(",")))

# Check if the audio file or model path exists
if args.audio and not os.path.exists(args.audio):
    parser.error(f"The specified audio ({args.audio}) file does not exist")

if not os.path.exists(args.model_path):
    parser.error("The specified model file does not exist")
else:
    model = tf.keras.models.load_model(args.model_path)
    print(f"Model `{args.model_path}` is loaded ")

if args.use_recorder:
    print(f"Using voice recorder")
    output_file = record_audio()
    prediction = process_audio(output_file)
    print(f"Prediction: {prediction}")

else:
    print(f"Using audio {args.audio}")
    prediction = process_audio(args.audio)
    print(f"Prediction: {prediction}")
