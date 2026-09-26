README.md


# 🌱 SoilVision AI

SoilVision is an AI-based soil classification system that uses
Convolutional Neural Networks (CNN) to classify soil images.

## Soil Classes

The model classifies seven soil types:

1. Alluvial Soil
2. Arid Soil
3. Black Soil
4. Laterite Soil
5. Mountain Soil
6. Red Soil
7. Yellow Soil

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

## Model

The CNN consists of:

- Conv2D - 32 filters
- MaxPooling
- Conv2D - 64 filters
- MaxPooling
- Flatten
- Dense - 128 neurons
- Softmax output - 7 classes

## Input

Images are resized to:

128 × 128 pixels

Pixel values are normalized from:

0–255 → 0–1

## Training

The dataset is divided into:

- 80% Training
- 20% Validation

The model is trained for 10 epochs.

## Running the Application

Run:

```bash
streamlit run app.py