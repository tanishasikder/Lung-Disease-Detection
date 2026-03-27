# Lung-Disease-Detection

## Overview

This project builds a **deep learning model to classify lung diseases** from chest X-ray images.

It uses:

* A **custom Convolutional Neural Network (CNN)** built with TensorFlow/Keras
* **Data augmentation** to improve generalization
* **Kaggle dataset integration** for automatic downloading

The model performs **multi-class classification** across **3 lung conditions**.

---

## How It Works

### 1. Dataset Loading

* Dataset is downloaded directly using the Kaggle API
* Images are loaded with:

  ```id="dl1"
  image_dataset_from_directory
  ```
* Automatically splits into:

  * Training set (90%)
  * Validation set (10%)

---

### 2. Data Augmentation (VERY important)

To prevent overfitting and make the model less dumb:

* Random horizontal flips
* Random rotations
* Random translations
* Random zoom

This helps the model generalize instead of memorizing images like a bot.

---

### 3. Model Architecture

A **deep CNN** with increasing complexity:

* Conv2D (16 filters)
* Conv2D (32 filters)
* Conv2D (64 filters × multiple layers)
* MaxPooling layers to reduce spatial size
* Dropout layers (0.2) to reduce overfitting
* Dense layer (512 units)
* Output layer (3 classes, softmax)

---

## Architecture Summary

| Layer Type | Purpose                                      |
| ---------- | -------------------------------------------- |
| Conv2D     | Extract features (edges, textures, patterns) |
| MaxPooling | Downsample image                             |
| Dropout    | Prevent overfitting                          |
| Dense      | Learn high-level patterns                    |
| Softmax    | Output class probabilities                   |

---

## Installation

```bash id="inst1"
pip install tensorflow kaggle matplotlib numpy
```

---

## Setup (Kaggle API)

1. Download your `kaggle.json` from Kaggle
2. Place it in:

```
~/.kaggle/kaggle.json
```

3. Run the script — dataset downloads automatically 🎯

---

## Running the Project

```bash id="run1"
python main.py
```

---

## Key Features

* End-to-end deep learning pipeline
* Automatic dataset download from Kaggle
* Built-in data augmentation
* Efficient data pipeline with:

  * `.cache()`
  * `.shuffle()`
  * `.prefetch()`
* Early stopping callback at **95% training accuracy**

---

## Training Details

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Epochs        | 10                       |
| Batch Size    | 128                      |
| Optimizer     | RMSprop                  |
| Learning Rate | 0.001                    |
| Loss Function | Categorical Crossentropy |

---

## Evaluation

The model tracks:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Visualization:

```id="plot1"
plot_performance(model_perform, "accuracy")
plot_performance(model_perform, "loss")
```

