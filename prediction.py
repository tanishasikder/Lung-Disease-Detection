import kagglehub
import tensorflow as tf
import random
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import pathlib

#need to find image size and fix environment issues

# Load in dataset and process the file
path = kagglehub.dataset_download("fatemehmehrparvar/lung-disease")
data_dir = tf.keras.utils.get_file(origin=path, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

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

#Data augmentation to help the model generalize better
data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    # Randomly flip some images
    tf.keras.layers.RandomFlip('horizontal'),
    # Randomly rotate some images
    tf.keras.layers.RandomRotation(0.2, full_mode='nearest'),
    # Move to the center of the image by a random amount
    tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
    # Zooming in some images
    tf.keras.layers.RandomZoom(0.2, fill_mode='nearest')
])

# Final model has the augmentation and the original model
final_model = tf.keras.models.Sequential([
    data_augmentation,
    model
])

final_model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# The initial train and validation datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(150, 150),
    batch_size=128,
    label_mode='categorical',
    validation_split=0.10,
    subset='training',
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(150, 150),
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

