import keras
import numpy as np
import librosa

from internal_methods import spectrogramFromFile


class SpectrogramGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self, list_IDs, label_map, sample_shape, batch_size=32, shuffle=True, sr=44100
    ):
        "Initialization"
        self.dim = sample_shape
        self.batch_size = batch_size
        self.labels = label_map
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.sr = sr
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.preprocess_sample(ID)

            # Store class
            y[i] = self.labels[ID]

        return X, y

    def preprocess_sample(self, audio_path: str):
        sr = self.sr
        return spectrogramFromFile(
            audio_filepath=audio_path,
            sr=sr,
            expand_last_dim=True,
            pre_emphasis_coef=0.95,
            use_normalization=True,
        )
