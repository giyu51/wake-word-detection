# Wake Word Detection 🎙️

This project introduces a real-time system capable of detecting wake words using a neural network model that processes 2-second audio spectrograms.

---

## Project Overview 🚀

The wake word detection system follows a streamlined workflow:

1. **Data Collection**: Background noise and wake word audio data were collected, supplemented by the UrbanSound8K dataset.
2. **Data Preprocessing**: Extensive data augmentation techniques were applied to increase the sample diversity.
3. **Generator Creation**: A specialized generator was developed to convert audio files into spectrograms, integrating optimizations such as batch normalization and preemphasis.
4. **Model Training**: A neural network model was trained using the preprocessed data and saved for future use.
5. **Real-Time Prediction**: A script was deployed to continuously capture 1-second audio chunks, with the last 2 seconds used for model prediction.

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

### Model Information

```
    +---------------------+
    |      Input          |
    +----------+----------+
               |
               v
    +---------------------+
    |   Conv2D (32 filters)|
    +----------+----------+
               |
               v
    +---------------------+
    |   Batch Normalization|
    +----------+----------+
               |
               v
    +---------------------+
    |       ReLU          |
    +----------+----------+
               |
               v
    +---------------------+
    |   MaxPooling2D      |
    +----------+----------+
               |
               v
    +---------------------+
    |   Residual Block 1  |
    +----------+----------+
               |
               v
    +---------------------+
    |   Residual Block 2  |
    +----------+----------+
               |
               v
    +---------------------+
    |   Residual Block 3  |
    +----------+----------+
               |
               v
    +---------------------+
    |   Residual Block 4  |
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
    |       Dropout       |
    +----------+----------+
               |
               v
    +---------------------+
    |        LSTM         |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dropout       |
    +----------+----------+
               |
               v
    +---------------------+
    |        LSTM         |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dropout       |
    +----------+----------+
               |
               v
    +---------------------+
    |        LSTM         |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dropout       |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dense         |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dropout       |
    +----------+----------+
               |
               v
    +---------------------+
    |       Dense         |
    +----------+----------+
               |
               v
    +---------------------+
    |      Output         |
    +---------------------+

```

### Additional Information 📄

Included are files for data gathering and preprocessing to provide insight into the applied audio preprocessing techniques.
