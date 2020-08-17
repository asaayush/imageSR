# This is a Image Super Resolution Netowrk to be applied eventually on videos.


# Import the necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import cv2
from keras.preprocessing import image           # For Image Pre-processing
import PIL
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.initializers import glorot_uniform
from tensorflow import train
import pydot
from pydot import print_function


# FUNCTIONS
# Import the video


# Extract the frames


# Import Supplementary Images for Increasing Training and Testing Data
def import_high_res(train_dir, batch, img_size, info=0):
    data_source_training = train_dir
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_training,
                                                                   label_mode=None,
                                                                   batch_size=batch,
                                                                   image_size=img_size,
                                                                   validation_split=0.2,
                                                                   subset='training',
                                                                   seed=44)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_training,
                                                                  label_mode=None,
                                                                  batch_size=batch,
                                                                  image_size=img_size,
                                                                  validation_split=0.2,
                                                                  subset='validation',
                                                                  seed=44)
    if info == 1:
        print(len(train_ds))
        print(type(train_ds))
        print(train_ds)
    return train_ds, test_ds


# Implement Downsizing/Down-sampling Function
def downsizing(data_set, size):
    temp = data_set.map(lambda img: tf.image.resize(img, size))
    return temp


# Create & Compile Model
def create_model(image_shape=(1280, 720)):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=image_shape),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ]
    )
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=image_shape),
        # Block 1
        layers.Conv2D(filters=64, kernel_size=(4, 4), strides=4, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),
        # Block 2
        layers.Conv2D(filters=256, kernel_size=(2, 2), strides=2, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),
        # Block 3
        layers.Conv2D(filters=512, kernel_size=(1, 1), strides=1, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),
        # Block 4
        layers.Conv2D(filters=1024, kernel_size=(5, 5), strides=5, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),
        # Block 5
        layers.Conv2D(filters=2048, kernel_size=(2, 2), strides=2, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),
        # Block 6
        layers.Conv2D(filters=2048, kernel_size=(5, 5), strides=1, kernel_initializer="glorot_uniform"),
        layers.BatchNormalization(axis=3),
        layers.Activation('relu'),

        layers.Flatten(),
        layers.UpSampling3D(size=(1920, 1080, 3)),
        # layers.Dense(units=, activation=None),
        # layers.Reshape((1920, 1080, 3))
    ])
    model.summary()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# Train Model
def fit_data(model, train_data, test_data, epochs):
    history = model.fit(train_data, validation_data=test_data, epochs=epochs, batch_size=1)
    return history

# Test Model Performance


# Visualize Performance
def visualizer(history, epochs):
    accuracy = history.history['accuracy']
    accuracy_val = history.history['val_accuracy']
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, accuracy_val, label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, loss_val, label='Validation Loss')
    plt.title('Loss')
    plt.legend(loc='upper left')
    plt.show()

# Use Low Resolution Videos to Generate High Resolution Videos Using Model


# ESPCN Performance
def espcn_model(image_shape):
    x_input = tf.keras.Input(image_shape)
    x = layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=image_shape)(x_input)
    x = layers.Conv2D(128, 5, activation='relu', strides=5, name='conv1')(x)
                      # kernel_initializer=glorot_uniform(seed=34))(x_input)
    x = layers.Conv2D(150, 4, activation='relu', name='conv2')(x)
                      # kernel_initializer=glorot_uniform(seed=65))
    x = layers.Conv2D(192, kernel_size=(14, 7), activation='relu', padding='valid')(x)
                      # name='conv3', kernel_initializer=glorot_uniform(seed=12))(x)
    x = tf.nn.depth_to_space(input=x, block_size=8, data_format='NHWC')
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])
    model.summary()
    return model


# MAIN CODE
train_directory = 'datasets/single_combined/training'
batch_size = 2
image_size = (1980, 1080)
down_sampled_shape = (1280, 720)

training_data_set, testing_data_set = import_high_res(train_directory, batch_size, image_size)
ds_train_data = downsizing(training_data_set, down_sampled_shape)
ds_test_data = downsizing(testing_data_set, down_sampled_shape)

# model1 = create_model((1280, 720, 3))
# perf = fit_data(model1, ds_train_data, ds_test_data, epochs=10)
# visualizer(perf, epochs=10)

ESPCN_model = espcn_model((1280, 720, 3))
perf = fit_data(ESPCN_model, ds_train_data, ds_test_data, epochs=10)

visualizer(perf, epochs=10)
