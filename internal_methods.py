import librosa
import librosa.display

import numpy as np
import os
import random

from typing import Iterable
from matplotlib import pyplot as plt
from soundfile import write
from multiprocessing import Pool


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


def spectrogramFromDir(
    directory_name: str,
    sample_count: None | int = None,
    shuffle: bool = True,
    sr: int = 44100,
    mono: bool = True,
    expand_last_dim: bool = False,
):
    audio_paths = [
        os.path.join(directory_name, audio_name)
        for audio_name in os.listdir(directory_name)
    ]

    if shuffle:
        random.shuffle(audio_paths)

    if sample_count is not None:
        audio_paths = audio_paths[:sample_count]

    spectrograms = []

    for _path in audio_paths:
        spectrogram_db = spectrogramFromFile(
            file_path=_path, sr=sr, mono=mono, expand_last_dim=expand_last_dim
        )
        spectrograms.append(spectrogram_db)

    return np.array(spectrograms)


def displaySpectrogram(_spectrogram: np.ndarray, sr: int = 44100) -> None:
    librosa.display.specshow(
        _spectrogram, sr=sr, x_axis="time", y_axis="mel"
    )  # Adjust parameters as needed
    plt.colorbar(format="%+2.f dB")  # Add colorbar with specific format
    plt.title("Spectrogram")
    plt.show()


def createShiftSequence(
    start: float | int, stop: float | int, step: float | int
) -> np.ndarray:

    decimal_points = len(str(step).split(".")[-1])

    def format_element(x):
        formatted_x = np.format_float_positional(
            x, precision=decimal_points, fractional=True
        )  # returns x as string
        return float(formatted_x)

    vectorized_format = np.vectorize(format_element)

    arr = np.arange(start, stop, step)

    arr = vectorized_format(arr)

    return arr


def _pitchAugmentation(audio_path, output_directory, pitch_shifts, sr):
    """
    Performs pitch augmentation on a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        output_directory (str): Path to the directory for storing pitch-shifted files.
        pitch_shifts (Iterable): An iterable containing pitch shifts in semitones.
        sr (int): Sampling rate of the audio file.
    """

    audio_name = os.path.basename(audio_path)
    file_name, file_ext = os.path.splitext(audio_name)

    for shift in pitch_shifts:
        y, _ = librosa.load(audio_path, sr=sr)  # Ignore sample rate in output
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=shift)

        output_file = f"{file_name}_pitch_{shift}{file_ext}"
        output_path = os.path.join(output_directory, output_file)
        write(output_path, y_shifted, sr)


def pitchAugmentation(
    input_directory: str,
    output_directory: str,
    pitch_shifts,
    sr: int = 44100,
    num_workers: int = 4,
):
    """
    Performs pitch augmentation on audio files in a directory using multiprocessing.

    Args:
        input_directory (str): Path to the directory containing audio files.
        output_directory (str): Path to the directory for storing pitch-shifted files.
        pitch_shifts (Iterable): An iterable containing pitch shifts in semitones.
        sr (int, optional): Sampling rate of audio files. Defaults to 44100.
        num_workers (int, optional): Number of worker processes to use. Defaults to 4.
    """

    # Validate output directory existence
    os.makedirs(output_directory, exist_ok=True)

    audio_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".wav")
    ]

    with Pool(processes=num_workers) as pool:  # Corrected closing parenthesis
        args = zip(
            audio_files,
            [output_directory] * len(audio_files),
            [pitch_shifts] * len(audio_files),
            [sr] * len(audio_files),
        )
        pool.starmap(
            _pitchAugmentation,
            args,
        )  # Apply pitch augmentation in parallel


def _volumeAugmentation(audio_path, output_directory, volume_shifts, sr):
    """
    Performs volume augmentation on a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        output_directory (str): Path to the directory for storing volume-augmented files.
        volume_shift (float): Volume shift value.
        sr (int): Sampling rate of the audio files.
    """

    audio_name = os.path.basename(audio_path)
    file_name, file_ext = os.path.splitext(audio_name)

    print(output_directory)
    for shift in volume_shifts:

        y, _ = librosa.load(audio_path, sr=sr)
        y_changed = y * shift
        output_file = f"{file_name}_volume_{shift}{file_ext}"
        output_path = os.path.join(output_directory, output_file)
        write(output_path, y_changed, sr)


def volumeAugmentation(
    input_directory: str,
    output_directory: str,
    volume_shifts: Iterable,
    sr: int = 44100,
    num_workers: int = 4,
):
    """
    Performs volume augmentation on audio files in a directory using multiprocessing.

    Args:
        input_directory (str): Path to the directory containing audio files.
        output_directory (str): Path to the directory for storing volume-augmented files.
        volume_shifts (Iterable): An iterable containing volume shift values.
        sr (int, optional): Sampling rate of the audio files. Defaults to 44100.
        num_workers (int, optional): Number of worker processes to use. Defaults to 4.
    """

    audio_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".wav")
    ]

    with Pool(processes=num_workers) as pool:
        # Create arguments for each audio file and volume shift combination
        args = zip(
            audio_files,
            [output_directory] * len(audio_files),
            [volume_shifts] * len(audio_files),
            [sr] * len(audio_files),
        )
        pool.starmap(
            _volumeAugmentation, args
        )  # Utilize starmap for parallel execution




def _noiseAugmentation(
    audio_path, output_directory, noise_factors: Iterable, sr=44100
):
    """
    Adds noise to an audio file and saves the result.

    Args:
        audio_path (str): Path to the input audio file.
        output_directory (str): Path to the directory for storing noisy audio files.
        noise_factor (float): Factor by which to scale the noise. Default is 0.005.
        sr (int): Sampling rate of the audio file. Default is 44100.
    """

    audio_name = os.path.basename(audio_path)
    file_name, file_ext = os.path.splitext(audio_name)

    for noise_factor in noise_factors:
        print(noise_factor)
        audio, sr = librosa.load(audio_path, sr=sr)
        # Generate noise
        noise = np.random.randn(len(audio))

        # Add noise to the audio
        audio_noisy = audio + noise_factor * noise

        audio_noisy = np.clip(audio_noisy, -1.0, 1.0)

        output_file = f"{file_name}_noise_{noise_factor}{file_ext}"
        output_path = os.path.join(output_directory, output_file)
        write(output_path, audio_noisy, sr)



def noiseAugmentation(
    input_directory: str,
    output_directory: str,
    noise_factors: Iterable,
    sr: int = 44100,
    num_workers: int = 4,
):
    audio_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".wav")
    ]

    with Pool(processes=num_workers) as pool:
        # Create arguments for each audio file and volume shift combination
        args = zip(
            audio_files,
            [output_directory] * len(audio_files),
            [noise_factors] * len(audio_files),
            [sr] * len(audio_files),
        )
        pool.starmap(
            _noiseAugmentation, args
        )  # Utilize starmap for parallel execution
