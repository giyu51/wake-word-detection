# wake-word-detection
A real-time system that detects wake words using a neural network model on 2-second audio spectrograms.



**Real-Time Wake Word Detection**

This project implements a real-time wake word detection system with the following workflow:

1. **Data Collection**: Gathered background noise and wake word audio data, supplemented with UrbanSound8K dataset.
2. **Data Preprocessing**: Performed data augmentation to significantly increase the number of samples.
3. **Generator Creation**: Developed a generator to convert audio files into spectrograms, incorporating optimizations like batch normalization and preemphasis.
4. **Model Training**: Trained a neural network model and saved it.
5. **Real-Time Prediction**: Deployed a script to continuously record audio in 1-second chunks, using the last 2 seconds for model prediction.

---

