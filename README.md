# Wake Word Detection 🎙️

This project introduces a real-time system that detects wake words using a neural network model processing 2-second audio spectrograms.

---

<p style="background-color: darkblue; color:white; padding: 15px 10px; border-radius: 5px; border:2px solid white">

# Portfolio Project 📁

This project showcases:

1. 📊 Data preprocessing (audio) using various techniques
2. 🛠️ Building and understanding model structures
3. 🔧 Creating and using custom blocks like residual blocks
4. 🧠 Understanding model architecture
5. 🚀 Techniques such as reducing learning rate on plateau, early stopping, and checkpoint saving
6. 🎧 Processing real-time data

This project is part of larger projects like a Home Voice Assistant.

</p>

## Project Overview 🚀

The core model for wake word detection is trained on the author's voice and performs well. The wake word is "Hey Kocho," a reference to Shinobu Kocho from the anime-manga series Demon Slayer.

The system follows a streamlined workflow:

1. **Data Collection**: 🗃️ Collected background noise and wake word audio data, supplemented by the UrbanSound8K dataset.
2. **Data Preprocessing**: 🛠️ Applied extensive data augmentation techniques to increase sample diversity.
3. **Generator Creation**: 🖼️ Developed a specialized generator to convert audio files into spectrograms, with optimizations like batch normalization and preemphasis.
4. **Model Training**: 🧠 Trained a neural network model with the preprocessed data and saved it for future use.
5. **Real-Time Prediction**: 🎧 Deployed a script to continuously capture 1-second audio chunks, using the last 2 seconds for model prediction.

The README.md will focus mainly on the last two parts - Model Training and Real-Time Prediction.

---

## :link: Dependencies

To run this project, you need to have the following dependencies installed:

- 🐍 [Python](https://www.python.org/downloads/): Python is a programming language used by this project.
- 📦 [pip](https://pip.pypa.io/en/stable/): A package manager for installing Python libraries and packages.
- 🧠 [TensorFlow](https://www.tensorflow.org/): An open-source machine learning framework used for building and training the model.
- 🔢 [NumPy](https://numpy.org/): A library for numerical computing in Python, used for handling arrays and data.
- 📈 [Matplotlib](https://matplotlib.org/): A plotting library for creating visualizations from data.
- 🎵 [Librosa](https://librosa.org/): A Python package for music and audio analysis, used for audio processing tasks.
- 🔊 [Soundfile](https://pysoundfile.readthedocs.io/en/latest/): A Python library for reading and writing sound files.
- 🎮 [Pygame](https://www.pygame.org/): Used for playing sounds.
- 🎤 [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/): Python bindings for PortAudio, used for audio input and output.

These libraries provide the necessary tools for building, training, and evaluating the model, as well as handling audio input and output, and visualizing the results.

---

## Usage Instructions 📝

To utilize the system, follow these steps:

1. **Obtain a Model**:

   - Download a pre-trained model (male voice only).
   - Fine-tune a pre-trained model with your dataset.
   - Use your custom pre-trained model. Ensure the input shapes match and the output is suitable for binary classification.

2. **Configure and Run**:
   - Configure the `Application.py` Python file according to your requirements.
   - Run the script, and you're ready to go!

---

## Model Training

#### Model Architecture:

```py
def residual_block(x, filters:int, kernel_size:int|Tuple[int]=3, strides:int|Tuple[int]=1, activation:str="relu",padding:str="same" ):

    y = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    y = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding=padding)(y)
    y = layers.BatchNormalization()(y)

    if x.shape[-1] != filters:
        # Use pointwise convolution to manipulate filter number without changing dimenstions of spatial data
        x = layers.Conv2D(filters=filters, kernel_size=1, strides=strides, padding=padding)(x)

    out = layers.Add()([x, y]) # Skip Connection
    out = layers.Activation(activation)(out)
    return out



def build_model(input_shape, batch_size=32):
    inputs = Input(shape=input_shape, batch_size=batch_size)

    x = layers.Conv2D(filters=32, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=256, strides=2)
    x = residual_block(x, filters=512, strides=2)

    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(
        units=256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(
        units=512, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(units=512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
```

💡Additionally, you can check ["Model Architecture Flowchart"](#model-architecture-flowchart)

---

#### Training:

Parameters:

> **Epochs**: 70

> **Batch size**: 32

> **_Model Parameters_**:
>
> > **Total**: 20,422,657
>
> > **Trainable**: 20,418,753
>
> > **Non-trainable**: 3,904

> **_Dataset_**:
>
> > **"Wake word" class (1)**: 224,400
>
> > **"Background" class (0)**: 224,400
>
> > **Train-test split**: 20%
>
> > **X-train samples** : 359040
>
> > **X-test samples** : 89760
>
> > **Starting LR** (Learning Rate): 1e-3 (0.001)

> **Loss**: Binary CrossEntropy

> **Optimizer**: SGD, momentum=0.9

> **Callbacks**: EarlyStopping, LR Reduction on Plateau, Checkpoint save

> **Input shape**: Spectrogram of shape (40, 173)

> **Training Time**: 48 hours

> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU

#### Evaluation:

> **_Evalution of test data_**:
>
> > **Loss**:
>
> > **Accuracy**: 1.0

#### Graphs:

**Loss over epochs**

![Loss]("images/loss.png")

<br>

**Accuracy over epochs**

![Accuracy]("images/accuracy.png")

#### Model Architecture Flowchart

```
    +---------------------+
    |        Input        |
    +----------+----------+
               |
               v
    +---------------------+
    |     Conv2D (32)     |
    +----------+----------+
               |
               v
    +---------------------+
    | Batch Normalization |
    +----------+----------+
               |
               v
    +---------------------+
    |        ReLU         |
    +----------+----------+
               |
               v
    +---------------------+
    |    MaxPooling2D     |
    +----------+----------+
               |
               v
    +---------------------+
    | Residual Block (64) |
    +----------+----------+
               |
               v
    +---------------------+
    | Residual Block (128)|
    +----------+----------+
               |
               v
    +---------------------+
    | Residual Block (256)|
    +----------+----------+
               |
               v
    +---------------------+
    | Residual Block (512)|
    +----------+----------+
               |
               v
    +---------------------+
    |   TimeDistributed   |
    |       Flatten       |
    +----------+----------+
               |
               v
    +---------------------+
    |     Dropout (0.3)   |
    +----------+----------+
               |
               v
    +---------------------+
    |      LSTM (512)     |
    +----------+----------+
               |
               v
    +---------------------+
    |     Dropout (0.3)   |
    +----------+----------+
               |
               v
    +---------------------+
    |      LSTM (512)     |
    +----------+----------+
               |
               v
    +---------------------+
    |     Dropout (0.3)   |
    +----------+----------+
               |
               v
    +---------------------+
    |      LSTM (512)     |
    +----------+----------+
               |
               v
    +---------------------+
    |     Dropout (0.3)   |
    +----------+----------+
               |
               v
    +---------------------+
    |      Dense (128)    |
    +----------+----------+
               |
               v
    +---------------------+
    |     Dropout (0.3)   |
    +----------+----------+
               |
               v
    +---------------------+
    |      Dense  (1)     |
    +----------+----------+
               |
               v
    +---------------------+
    |       Output        |
    +---------------------+

```

### Additional Information 📄

Included are files for data gathering and preprocessing to provide insight into the applied audio preprocessing techniques.
