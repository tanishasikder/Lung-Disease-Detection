import kagglehub
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Load in dataset and process the file
num_epochs = 10

path = "fatemehmehrparvar/lung-disease"
api = KaggleApi()
api.authenticate()
api.dataset_download_files(path, path="data", unzip=True)
data_dir = "data/Lung X-Ray Image"

#Data augmentation to help the model generalize better
data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape=(300, 300, 3)),
    # Randomly flip some images
    tf.keras.layers.RandomFlip('horizontal'),
    # Randomly rotate some images
    tf.keras.layers.RandomRotation(0.2, fill_mode='nearest'),
    # Move to the center of the image by a random amount
    tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
    # Zooming in some images
    tf.keras.layers.RandomZoom(0.2, fill_mode='nearest')
])

# The initial train and validation datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(300, 300),
    batch_size=128,
    label_mode='categorical',
    validation_split=0.10,
    subset='training',
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(300, 300),
    batch_size=128,
    label_mode='categorical',
    validation_split=0.10,
    subset='validation',
    seed=42
)

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

# Final train and validation datasets
train = (train_dataset
         .cache()
         .shuffle(SHUFFLE_BUFFER_SIZE)
         .prefetch(PREFETCH_BUFFER_SIZE))

validation = (validation_dataset
              .cache()
              .shuffle(SHUFFLE_BUFFER_SIZE)
              .prefetch(PREFETCH_BUFFER_SIZE))

# Create the model. It is for multiclass classification
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(300, 300, 3)),
    tf.keras.layers.Rescaling(1./255), #Rescaling layer
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2), #Dropout layer to prevent overfittin

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Final model has the augmentation and the original model
final_model = tf.keras.models.Sequential([
    data_augmentation,
    model
])

final_model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

final_model.fit(train, epochs=num_epochs, validation_data=validation, verbose=2)
