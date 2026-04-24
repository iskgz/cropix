# Cropix

Cropix is a wheat disease analysis project that combines computer vision and NLP to support diagnosis from two different inputs:

- an image of the plant
- a text description of symptoms

The goal is to compare the two signals, produce a final disease prediction, and keep the workflow simple enough to run and evaluate locally.

## What the project does

- Trains an image classification model for wheat disease recognition
- Trains an NLP classifier from symptom descriptions
- Runs inference from either modality
- Combines image and text predictions into a final decision
- Exposes reusable scripts for training, prediction, and TFLite testing

## Repository scope

This repository contains the application and model code required to reproduce the workflow.

The raw dataset is intentionally not included in the public repository.

## Project structure

```text
cropix/
├── main.py
├── model/
│   ├── train.py
│   ├── predict.py
│   ├── test_tflite.py
│   └── model artifacts
├── nlp/
│   ├── train_nlp.py
│   ├── predict_nlp.py
│   ├── interactive_nlp.py
│   └── NLP artifacts
└── README.md
```

## Architecture

### 1. Image pipeline

The image pipeline focuses on supervised classification of wheat leaf or spike images.

Typical flow:

1. Load labeled images from the training dataset
2. Preprocess the images
3. Train a CNN-based classifier using transfer learning
4. Export the trained model
5. Run local inference with the saved model or TFLite version

### 2. NLP pipeline

The NLP pipeline works on symptom text entered by the user.

Typical flow:

1. Clean and normalize the symptom text
2. Convert text into numerical features
3. Train a text classifier
4. Save the vectorizer and model
5. Run inference on new symptom descriptions

### 3. Decision fusion

The final decision layer combines the image prediction and the NLP prediction.

The intent is to:

- keep the image model as the primary signal when a clear visual match exists
- use NLP as a complementary signal when text is available
- produce a final label with a confidence score that is easy to inspect

## Main labels

The project currently works with wheat-related disease classes such as:

- wheat_healthy
- wheat_loosesmut
- wheat_powderymildew
- wheat_rust
- wheat_septoria

## Installation

This project is designed for Python 3.12+ and a local development environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy pandas scikit-learn joblib opencv-python
```

## Training

### Train the NLP model

```bash
cd nlp
python train_nlp.py
```

### Train the image model

```bash
cd model
python train.py
```

## Inference

### NLP inference

```bash
cd nlp
python predict_nlp.py
```

### Image inference

```bash
cd model
python predict.py <image_path>
```

### TFLite validation

```bash
cd model
python test_tflite.py
```

## Data policy

- Raw datasets are not published here
- Large local training exports should stay out of version control
- Only the code and model artifacts needed to run the project should be committed

## Notes

- If you are evaluating the project, start with `main.py`
- If you want to inspect model behavior, start with `model/train.py` and `nlp/train_nlp.py`
- If you want to test the final inference flow, use the prediction scripts in `model/` and `nlp/`
