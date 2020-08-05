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
import glob

# FUNCTIONS
# Import the video


# Extract the frames


# Import Supplementary Images for Increasing Training and Testing Data
def importHighRes(train_dir, batch, img_size, info=0):
    data_source_training = train_dir
    # data_source_validation = 'datasets/single_combined/validation'

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_training,
                                                                   label_mode=None,
                                                                   batch_size=batch,
                                                                   image_size=img_size)
    # test_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_validation,
    #                                                              label_mode=None,
    #                                                              batch_size=1,
    #                                                              image_size=(3840, 2160))
    if info == 1:
        print(len(train_ds))
        print(type(train_ds))
        print(train_ds)


# Implement Downsizing/Down-sampling Function


# Create & Compile Model


# Train Model


# Test Model Performance


# Visualize Performance


# Use Low Resolution Videos to Generate High Resolution Videos Using Model


# MAIN CODE
train_directory = 'datasets/single_combined/training'
batch_size = 16
image_size = (3840, 2160)
importHighRes(train_directory, batch_size, image_size, info=1)
