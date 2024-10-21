import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

EPOCHS = 1
CHANNELS = 3
BATCH_SIZE = 32
IMAGE_SIZE = 256

if not os.path.exists("dataset"):
    print("Dataset directory not found!")
    exit(1)

try:
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

class_names = dataset.class_names
print("Available classes:", class_names)

def split_dataset(dataset, train_split, val_split, test_split):
    assert (train_split + val_split + test_split) == 1
    dataset_size = len(dataset)

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)

    return train_dataset, val_dataset, test_dataset

train_split, val_split, test_split = 0.8, 0.1, 0.1
train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_split, val_split, test_split)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
total_class = len(class_names)

model = models.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(total_class, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model_score = model.evaluate(test_dataset)
print('Model score:', model_score)

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

model.save("saved_models/model.keras")  # Save with appropriate extension

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
