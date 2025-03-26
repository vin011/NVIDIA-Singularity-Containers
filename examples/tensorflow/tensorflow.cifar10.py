#!/usr/bin/env python3
#
# Train a simple Convolutional Neural Network (CNN) to classify CIFAR images.
#
# https://www.tensorflow.org/tutorials/images/cnn
# https://touren.github.io/2016/05/31/Image-Classification-CIFAR10.html
# https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79

import os
import tensorflow as tf

# Download training and test image datasets
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Verify training and ...
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)
# ... test image dataset sizes
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)

# Normalize the 8-bit (3-channel) RGB image pixel data between 0.0 and 1.0
# https://en.wikipedia.org/wiki/8-bit_color
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model and its network architecture. A Sequential model is 
# appropriate for a network with a plain stack of layers, where each 
# layer has exactly one input tensor and one output tensor.
# https://www.tensorflow.org/guide/keras/sequential_model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10),
])

# Print the summary of the model's network architecture
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary
model.summary()

# Specify an optimizer, a loss function, and metrics, then compile the model.
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train the model
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
model.fit(
    x=x_train,
    y=y_train,
    batch_size=256,
    epochs=50,
    validation_split=0.2,
    verbose=2,
)

# Evaluate the model and its accuracy
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#evaluate
model.evaluate(
    x=x_test,
    y=y_test,
    batch_size=256,
    verbose=2,
)

# Save the model
model.save('saved_model.o')

# Source:
# Expanse:/cm/shared/examples/sdsc/tensorflow/tf2-train-cnn-cifar-10.py
