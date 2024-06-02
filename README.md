# 🔊 Wake Word Detection: "Hey Kocho"

## Portfolio Project 📁

This project was developed as part of a portfolio to demonstrate advanced skills in:

- 🎧 Audio processing and data augmentation
- 🏗️ Developing suitable model architectures
- 🔄 Implementing Residual Blocks followed by LSTMs
- 🧠 Using TensorFlow and Keras for model building
- ⚙️ Utilizing callbacks for training optimization (e.g., EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- 📊 Handling large datasets (448K+ samples)
- ⏱️ Training models over extended periods (72+ hours)

It showcases the ability to create a precise and robust model capable of differentiating between very similar sounding words.

---

## Overview

This project is a highly precise wake word detection system capable of distinguishing the wake word "Hey Kocho" from very similar sounding words like "Hey Couch" or "Hey Coach". This system was developed as a portfolio project to demonstrate advanced techniques in audio processing, data augmentation, and deep learning.


## Features

- **High Precision**: Detects "Hey Kocho" with minimal false positives.
- **Robustness**: Does not trigger on similar sounding words.
- **Large Dataset**: Trained on a dataset of 448,800 samples.
- **Advanced Model Architecture**: Utilizes Residual Blocks followed by LSTMs.
- **Extensive Training**: Model trained for over 72 hours.

## Table of Contents

- [🔊 Wake Word Detection: "Hey Kocho"](#-wake-word-detection-hey-kocho)
  - [Portfolio Project 📁](#portfolio-project-)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Background 📚](#background-)
  - [Dataset 📊](#dataset-)
  - [Audio Preprocessing](#audio-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Results](#results)
  - [Important Note ⚠️](#important-note-️)
  - [Usage](#usage)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Background 📚

Wake word detection is a crucial component in many voice-activated systems, enabling the system to remain in a low-power state until activated by a specific phrase. This project aims to push the boundaries of precision in wake word detection using advanced neural network architectures.

The wake word "Hey Kocho" is inspired by Shinobu Kocho from the anime-manga series Demon Slayer (Kimetsu no Yaiba). 🦋

<img src="Shinobu_Kocho.png" style="width: 20%;">


## Dataset 📊

The dataset for this project was created and augmented to include 225K samples. This large dataset ensures the model is trained on a variety of voices and acoustic conditions.

## Audio Preprocessing

Detailed audio preprocessing steps were implemented to ensure high-quality input data:

- Noise reduction
- Normalization
- Data augmentation techniques such as pitch shifting, time stretching, and adding background noise

## Model Architecture

The model architecture consists of:

- **Residual Blocks**: Capture and enhance features from the input audio.
- **LSTMs (Long Short-Term Memory)**: Capture temporal dependencies in the audio signals.

## Training

- **Training Duration**: Over 72 hours
- **Optimizer**: [Specify the optimizer used, e.g., Adam]
- **Loss Function**: [Specify the loss function used, e.g., Binary Cross-Entropy]
- **Evaluation Metrics**: [Specify the metrics used, e.g., Accuracy, Precision, Recall]

## Results

[Provide details on the performance of the model. Include metrics, confusion matrix, or any relevant graphs that demonstrate the model's precision and robustness.]


## Important Note ⚠️

The model is trained only on the voice of the author. It may not perform optimally with other voices and should be fine-tuned with additional data for broader usage.


## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/wake-word-detection.git
   cd wake-word-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the model**:
   `bash
python run_model.py --input your_audio_file.wav
`
   [Provide detailed usage instructions, including examples and explanations of input/output.]

## Installation

[Include any additional installation steps, such as setting up a virtual environment, downloading pretrained models, etc.]

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

[Specify the license under which the project is distributed, e.g., MIT License.]

## Acknowledgements

- [Acknowledge any individuals, libraries, or resources that were instrumental in the development of the project.]

---
