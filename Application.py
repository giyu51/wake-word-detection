
import pyaudio
import numpy as np
from collections import deque
import soundfile as sf
import tensorflow as tf
import time
from internal_methods import spectrogramFromAudioData
import pygame
import os
from typing import Any

response_mapping = {
    "wake_response_audio":"kocho_sounds/moshi_moshi.wav", "wake_response_message":"Moshi moshi", "greeting_message":"Konnichiwa", "greeting_audio":"kocho_sounds/konnichiwa.wav", "goodbye_audio":"kocho_sounds/sayonara.wav", "goodbye_message":"Sayonaraa..."
}

class AudioProcessor:
    def __init__(
        self,
        sample_shape: tuple | None = None,
        model_path: str = "model.keras",
        response_mapping: dict = {},
        use_model_warm_up=True,
        batch_size=32,
        rate=44100,
        channels=1,
        format=pyaudio.paFloat32,
        duration=2,
        save_dir="wake_record",
        verbose: int | bool = 1,
    ):
        # Audio related variables
        self.RATE: int = rate
        self.CHANNELS: int = channels
        self.FORMAT = format
        self.SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(format)
        self.DURATION: int = duration
        self.CHUNK: int = self.RATE  # 1 second chunks
        self.audio_buffer = deque(maxlen=self.DURATION)

        # Add a single chunk to the buffer, so that from the beginning of recording there will be already 2 chunks
        self.buffer_placeholder: np.ndarray = np.zeros(shape=(self.RATE,))
        self.audio_buffer.append(self.buffer_placeholder)

        # Saving related variables
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._saving_counter: int = len(os.listdir(self.save_dir))
        self.verbose = verbose

        # Model related variables
        self.sample_shape = sample_shape
        if self.sample_shape is None:
            self.sample_shape = self._define_sample_shape()
        self.batch_size: int = batch_size
        self.batch_placeholder: np.ndarray = np.zeros(
            shape=(self.batch_size - 1,) + self.sample_shape
        )  # Batch placeholder, So if the model is trained on batch of 32, it creates a shape of 31 samples and during prediction adds a single sample to the batch,
        # thus mathing the batch size

        self.model_path: str = model_path
        self.model = self.load_model()

        # Play beep sound everytime the model recognized a wake word

        self.mixer = pygame.mixer
        self.mixer.init()
        self.wake_response_channel = self.mixer.Channel(0)
        self.response_mapping = response_mapping
        self._load_responses()

        if use_model_warm_up:
            self.warm_up_model()

    def _load_responses(self):

        ### Wake Response

        self.wake_response_audio: str = self.response_mapping.get(
            "wake_response_audio", None
        )
        self.wake_response_message: str = self.response_mapping.get(
            "wake_response_message", None
        )
        self.wake_response_sound = self._get_sound_obj(self.wake_response_audio)

        ### Greetings

        self.greeting_audio: str = self.response_mapping.get("greeting_audio", None)
        self.greeting_message: str = self.response_mapping.get("greeting_message", None)
        self.greeting_sound = self._get_sound_obj(self.greeting_audio)

        ### Goodbye

        self.goodbye_audio: str = self.response_mapping.get("goodbye_audio", None)
        self.goodbye_message: str = self.response_mapping.get("goodbye_message", None)
        self.goodbye_sound = self._get_sound_obj(self.goodbye_audio)

    def _get_sound_obj(self, audio_name):
        return self.mixer.Sound(audio_name) if audio_name is not None else None

    def send_voice(self, sound, message=None):
        if sound is None:
            return
        self.wake_response_channel.play(sound)  # Play the beep sound on the channel

        isFinished = False
        while self.wake_response_channel.get_busy():
            if not isFinished and message is not None:
                print(self.send_message(message))
                isFinished = True

    def send_message(self, message):
        if message is not None:
            return message

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def warm_up_model(self, warm_up_cycles: int = 5):
        self.log("Warming up the model...")

        dummy_input = np.zeros(
            shape=(self.batch_size,) + self.sample_shape
        )  # Replace with appropriate shape
        for _ in range(warm_up_cycles):
            self.model.predict(dummy_input, verbose=0)

        self.log("Model warm-up completed.")

    def _define_sample_shape(self):
        buffer_shape = self.DURATION * self.RATE
        dummy_sequence = np.ones(shape=buffer_shape)
        spectrogram_sample = self._spectrogram_from_audio_data(dummy_sequence)
        sample_shape = spectrogram_sample.shape
        self.log(f"Sample shape defined: {sample_shape}")
        return sample_shape

    def _save_segments_to_wav(self, segments):
        combined_segments = np.concatenate(segments)
        filename = f"output_{self._saving_counter}.wav"
        save_path = os.path.join(self.save_dir, filename)
        sf.write(save_path, combined_segments, self.RATE, "FLOAT")
        self.log(f"Saved combined segments to {filename}")
        self._saving_counter += 1

    def _spectrogram_from_audio_data(self, sequence):
        return spectrogramFromAudioData(audio_data=sequence)

    def _predict(self, segments):
        sequence = np.concatenate(segments)
        spectrogram = self._spectrogram_from_audio_data(sequence)
        data_batch = np.vstack((self.batch_placeholder, spectrogram[np.newaxis, :, :]))
        prediction = self.model.predict(data_batch, verbose=0)
        prediction = prediction[-1][0]
        prediction = np.round(prediction, 2)
        return prediction

    def callback(self, in_data, frame_count, time_info, status):
        start_time = time.perf_counter()
        audio_segment = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.append(audio_segment)

        prediction = self._predict(self.audio_buffer)
        if prediction > 0.8:
            self.send_voice(
                sound=self.wake_response_sound, message=self.wake_response_message
            )
            self._save_segments_to_wav(
                self.audio_buffer
            )  # Uncomment if you want to save recognized wake word segments

        else:
            escalated_time = time.perf_counter() - start_time
            self.log(
                f"LISTENING... | Prediction: {prediction} | Time Escalated: {escalated_time:.2f}\n"
            )
            if escalated_time > self.DURATION:
                self.log(
                    f"WARNING: callback function took {escalated_time:.2f} seconds, which is longer than the chunk duration of 1 second.\n"
                )
        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.callback,
        )

        self.log("Recording...")
        stream.start_stream()

        self.send_voice(sound=self.greeting_sound, message=self.greeting_message)
        try:
            while stream.is_active():
                time.sleep(1)
        except KeyboardInterrupt:
            self.send_voice(self.goodbye_sound, message=self.goodbye_message)
            self.log("Interrupted by user")

        stream.stop_stream()
        stream.close()
        p.terminate()
        self.log("Recording stopped.")


if __name__ == "__main__":
    audio_processor = AudioProcessor(
        model_path="best_model.h5", response_mapping=response_mapping
    )
    audio_processor.start_stream()


