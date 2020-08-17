import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras import layers

# Important Note about Resolutions
#
#           ENCODER             |            DECODER
#                               |
# 2560x1440                     |                       2560x1440
#    1920x1080                  |                    1920x1080
#       1600x900                |                1600x900
#           1366x768            |            1366x768
#               1280x720        |        1280x720
#                   640x360     |     640x360
#                     320x180   |   320x180
#                       160x90  |  160x90
#                         80x45 | 80x45
#                             16x9
#
# This is the central concept of the U-Net Model I am trying to Implement.
# This will use the following concepts:
#       1) Skip Connections
#           ==> The layers with the same resolution are connected.
#       2) Fractional Stride Convolution Layers or Transposed Convolution Layers
#           ==> The process of increasing the resolution of the previous layer.
#       3) Convolutional Auto-encoder Design
#           ==> The process of the first half and second half of the above shape.
#
# Layers to be used:
#     1) layers.Conv2D()                  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
#     2) layers.BatchNormalization()      https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
#     3) layers.Conv2DTranspose()         https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
#     4) layers.Dense()                   https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
#     5) layers.Reshape()                 https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
#     6) layers.Add()                     https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add
#     7) layers.ZeroPadding2D()           https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
#     8) layers.Activation()              https://www.tensorflow.org/api_docs/python/tf/keras/layers/Activation
#     9) layers.MaxPooling2D()            https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D


def conv_block(x, filters, f, st=1):
    f1, f2, f3 = filters
    x_short = x
    # First block on main path
    x = layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=st, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Second block on main path
    x = layers.Conv2D(filters=f2, kernel_size=(f, f), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Third Block on main path
    x = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=1)(x)
    x = layers.BatchNormalization()(x)
    # Shortcut Block
    x_short = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=st, padding='valid')(x_short)
    x = layers.Add()([x, x_short])
    x = layers.Activation('relu')(x)
    return x


def identity_block(x, filters, f):
    f1, f2, f3 = filters
    x_short = x
    # First block on main path
    x = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Second block on main path
    x = layers.Conv2D(filters=f2, kernel_size=(f, f), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Third Block on main path
    x = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=1)(x)
    x = layers.BatchNormalization()(x)
    # Shortcut Block
    x = layers.Add()([x, x_short])
    x = layers.Activation('relu')(x)
    return x


def residual_block(x_in, filters, f, s=1):
    x = conv_block(x_in, filters=filters, st=s, f=f)
    x = identity_block(x, filters=filters, f=f)
    x = identity_block(x, filters=filters, f=f)
    x = identity_block(x, filters=filters, f=f)
    return x


def trans_conv_block(x, skip, s):
    x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(s, s), padding='same')(x)
    x = layers.Add()([x, skip])
    x = layers.Activation('relu')(x)
    x = residual_block(x, filters=[32, 32, 32], f=1, s=1)
    return x


def get_output(x, req_res):
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(15, 29), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(15, 29), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 29), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_out_1 = x_dec
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(19, 33), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(25, 43), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_out_2 = x_dec
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 53), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 53), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 55), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 55), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 55), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(31, 55), strides=(1, 1), padding='valid')(x_dec)
    x_dec = layers.Activation('relu')(x_dec)
    x_out_3 = x_dec
    x_dec = layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x_dec = layers.Activation('relu', dtype='float32')(x_dec)
    x_out_4 = x_dec

    if req_res == 1:
        # Target Res ==> 1366 x 768 x 3
        x_out = x_out_1
    elif req_res == 2:
        # Target Res ==> 1600 x 900 x 3
        x_out = x_out_2
    elif req_res == 3:
        # Target Res ==> 1920 x 1080 x 3
        x_out = x_out_3
    elif req_res == 4:
        # Target Res ==> 2560 x 1440 x 3
        x_out = x_out_4
    else:
        x_out = 0
    return x_out


def u_net_model(init_shape, final_size, lr_rate, req_result):
    x_input = layers.Input(init_shape)
    # Currently you have a 720p set of images. Let's rescale
    x_rescale = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(x_input)
    skip_5 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=1)(x_rescale)

    # ENCODING
    # For the Main Path, we have to go from 720p to 360.
    x = residual_block(x_rescale, filters=[32, 32, 32], f=3, s=2)
    skip_4 = x
    # From 360 we need to go again to 180
    x = residual_block(x, filters=[32, 32, 32], f=3, s=2)
    skip_3 = x
    # From 180 to 90
    x = residual_block(x, filters=[32, 32, 32], f=3, s=2)
    skip_2 = x
    # From 90 to 45
    x = residual_block(x, filters=[32, 32, 32], f=3, s=2)
    skip_1 = x
    # From 45 to 9
    x = residual_block(x, filters=[16, 16, 16], f=3, s=5)
    skip_0 = x

    # FLATTEN AND DENSE LAYERS
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.Dense(36)(x)

    # DECODING
    # Currently the shape is a flat 36
    x = layers.Reshape(target_shape=(9, 4, 1))(x)
    x = residual_block(x, filters=[64, 64, 64], f=1, s=1)
    x = layers.Reshape(target_shape=(9, 16, 16))(x)
    x = layers.Add()([x, skip_0])
    x = layers.Activation('relu')(x)
    # Now from 9x16x16 we need to keep up-scaling using transposed convolution
    x = trans_conv_block(x, skip_1, s=5)
    # From 45x80 to 90x160
    x = trans_conv_block(x, skip_2, s=2)
    # From 90x160 to 180x320
    x = trans_conv_block(x, skip_3, s=2)
    # From 180x320 to 360x640
    x = trans_conv_block(x, skip_4, s=2)
    # From 360x640 to 720x1280
    x = trans_conv_block(x, skip_5, s=2)

    # GOING BEYOND RECONSTRUCTION
    x_out = get_output(x, req_result)
    if req_result > 3:
        x_out = tf.image.resize(x_out, size=final_size, preserve_aspect_ratio=True)

    x_ups = tf.image.resize(x_rescale, size=final_size, method=tf.image.ResizeMethod.BICUBIC,
                            preserve_aspect_ratio=True)
    x_out = layers.Add(dtype='float32')([x_out, x_ups])
    x_out = layers.Activation('relu', dtype='float32')(x_out)

    # Compile and view summary
    model = tf.keras.Model(inputs=x_input, outputs=x_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate), loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    return model
