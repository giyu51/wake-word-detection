# Wake Word Detection 🎙️

This project introduces a real-time system that detects wake words using a neural network model processing 2-second audio spectrograms.

---

<div style="background-color: darkblue; color:white; padding: 15px 10px; border-radius: 5px; border:2px solid white">

# Portfolio Project 📁

This project showcases:

1. 📊 Data preprocessing (audio) using various techniques
2. 🛠️ Building and understanding model structures
3. 🔧 Creating and using custom blocks like residual blocks
4. 🧠 Understanding model architecture
5. 🚀 Techniques such as reducing learning rate on plateau, early stopping, and checkpoint saving
6. 🎧 Processing real-time data

This project is part of larger projects like a Home Voice Assistant.

</div>

## Project Overview 🚀

The core model for wake word detection is trained on the author's voice and performs well. The wake word is "Hey Kocho," a reference to [Shinobu Kocho](https://kimetsu-no-yaiba.fandom.com/wiki/Shinobu_Kocho) from the anime-manga series Demon Slayer.

The system follows a streamlined workflow:

1. **Data Collection**: 🗃️ Collected background noise and wake word audio data, supplemented by the UrbanSound8K dataset.
2. **Data Preprocessing**: 🛠️ Applied extensive data augmentation techniques to increase sample diversity.
3. **Generator Creation**: 🖼️ Developed a specialized generator to convert audio files into spectrograms, with optimizations like batch normalization and preemphasis.
4. **Model Training**: 🧠 Trained a neural network model with the preprocessed data and saved it for future use.
5. **Real-Time Prediction**: 🎧 Deployed a script to continuously capture 1-second audio chunks, using the last 2 seconds for model prediction.

The README.md will focus mainly on the last two parts - Model Training and Real-Time Prediction.

---

## 🔗 Dependencies

The link to the **PreTrained Model** (on voice of the author): https://drive.google.com/file/d/1Q_2NUY1HP67XyWa50pP_0NaSwIm4M2O9/view?usp=sharing

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

## Files 📂

1. **DataGathering.py** 🎙️: Used to gather data by recording the wake word or continuously recording background sounds.
2. **Preprocessing.ipynb** 🔄: Used for preprocessing audio, applying different audio augmentation techniques, and moving the augmented audios to a separate folder.
3. **KerasGenerator.py** ⚙️: A custom module for Keras generator creation.
4. **Training.ipynb** 🏋️: Used to assemble the dataset, train the model, and save it.
5. **Application.py** 🚀: The actual application integrating the trained model.

## Dataset Info 📊

The dataset consists of 2-second audio spectrograms.

In the current dataset, we used:

1. **"Wake word" class (1)**:

   - Recording of the author saying the wake word.
2. **"Background" class (0)**:

   - **Background**: Recordings of the author's background.
   - **Talk**: Recording of the author speaking and talking about everything except the wake word.
   - **Urban**: [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?resource=download&select=fold5) Dataset from Kaggle.

The dataset was split as follows:

- Wake samples = 50%
- Background samples = 50%

In background samples:

- Urban samples (2 seconds) = 15,082
- Background samples = (Number of wake samples - Number of urban samples) / 2
- Talk samples = Background samples

Note ⚠️: In the UrbanSound8K dataset, there were audio files of different durations, like 2 or 5 seconds. Therefore, we implemented a preprocessing function to extract each 2-second segment from all files.

### Model Architecture 🏗️

This section covers the code used for model building. The general structure involves acquiring features from spectrograms and feeding those features to RNN layers to learn the sequence and how to distinguish them.

In the CNN part, we start with simple CNN layers followed by normalization, activation, and pooling layers. Subsequently, we use Residual blocks, which proved to work even better than traditional convolutional layers due to the "skip connection."

💡 Additionally, you can check the [&#34;Model Architecture Flowchart&#34;](#model-architecture-flowchart)

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

---

## Training:

Parameters:

> **Epochs**: 70 ( Early Stopping on epoch 58)

> **Batch size**: 32

> **_Model Parameters_**:
>
>> **Total**: 20,422,657
>>
>
>> **Trainable**: 20,418,753
>>
>
>> **Non-trainable**: 3,904
>>

> **_Dataset_**:
>
>> **"Wake word" class (1)**: 224,400
>>
>
>> **"Background" class (0)**: 224,400
>>
>
>> **Train-test split**: 20%
>>
>
>> **X-train samples** : 359,040
>>
>
>> **X-test samples** : 89,760
>>
>
>> **Starting LR** (Learning Rate): 1e-3 (0.001)
>>

> **Loss**: Binary CrossEntropy

> **Optimizer**: SGD, momentum=0.9

> **Callbacks**: EarlyStopping, LR Reduction on Plateau, Checkpoint save

> **Input shape**: Spectrogram of shape (40, 173)

> **Training Time**: 48 hours

> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU

#### Evaluation:

> **_Evalution of test data_**:
>
>> **Loss**: 7e-4 (0.0007017801981419325)
>>
>
>> **Accuracy**: 1.0
>>

#### Graphs:

**Loss over epochs**

![Loss]()

<br>

**Accuracy over epochs**

![Accuracy]()

## Application 🎛️

The application is a class that handles all functions accordingly:

1. **Buffer Creation**: Creates a buffer using `deque` with a length of 2 to store two 1-second sequences.
2. **Stream Opening**: Opens a stream where each chunk is 1 second in duration, matching the model's training shape. For example, if the model was trained on spectrograms from audio with shape (88200,), each chunk should be (44100,) since the model was trained on 2-second audios.
3. **Chunk Recording**: Records a chunk and pushes it to the buffer.
4. **Callback Execution**: Sends the buffer to the model for prediction.
5. **Prediction Handling**: Manages the predictions from the model.

### Additional Details 📝

1. **Sound Playback**: The application can play sounds based on different conditions using a `response_mapping` dictionary passed during object creation. For more information, refer to the code.
2. **Initial Buffer State Handling**: Ensures the buffer works for 2 seconds initially to get 2 samples. This is achieved by creating an empty sequence at the initialization state so that the buffer size is always 2 during processing. This eliminates the need for the condition `if len(buffer) == DURATION`, which slows down the callback.
3. **Callback Warning**: Indicates a warning if the processing time exceeds the duration of a chunk, which can slow down the entire process.

## Additional Information 📄

Included are files for data gathering and preprocessing to provide insight into the applied audio preprocessing techniques.

## Model Architecture Flowchart

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
