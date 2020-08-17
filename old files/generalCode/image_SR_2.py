import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt

tf.autograph.set_verbosity(2)


def sr1_model(init_shape, final_shape):
    x_input = tf.keras.Input(init_shape)
    x = layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=init_shape)(x_input)
    x = layers.Conv2D(128, 5, activation='relu', strides=5, name='conv1')(x)
    x = layers.Conv2D(150, 4, activation='relu', name='conv2')(x)
    x = layers.Conv2D(192, kernel_size=(7, 14), activation='relu', padding='valid')(x)
    x = tf.nn.depth_to_space(input=x, block_size=8, data_format='NHWC')
    x = layers.experimental.preprocessing.Rescaling(1. * 255, input_shape=(final_shape))(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    model.summary()
    return model


def import_data(directory, batch):
    hr_image = (1080, 1920)
    lr_image = (720, 1280)
    hr_tr_data = tf.keras.preprocessing.image_dataset_from_directory(directory=directory + str('/hr/training'),
                                                                     image_size=hr_image,
                                                                     batch_size=batch,
                                                                     label_mode=None)
    hr_te_data = tf.keras.preprocessing.image_dataset_from_directory(directory=directory + str('/hr/validation'),
                                                                     image_size=hr_image,
                                                                     batch_size=batch,
                                                                     label_mode=None)
    lr_tr_data = tf.keras.preprocessing.image_dataset_from_directory(directory=directory + str('/lr/training'),
                                                                     image_size=lr_image,
                                                                     batch_size=batch,
                                                                     label_mode=None)
    lr_te_data = tf.keras.preprocessing.image_dataset_from_directory(directory=directory + str('/lr/validation'),
                                                                     image_size=lr_image,
                                                                     batch_size=batch,
                                                                     label_mode=None)
    # data_set = {'train_x': lr_tr_data,
    #            'train_y': hr_tr_data,
    #            'test_x': lr_te_data,
    #            'test_y': hr_te_data}
    return lr_tr_data, hr_tr_data, lr_te_data, hr_te_data


def fit_data(model, train_data, test_data, epochs, batch):
    history = model.fit(train_data, validation_data=test_data, epochs=epochs, batch_size=batch)
    return history


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


source = 'datasets/single_combined'
batch_size = 8
epch = 10
train_x, train_y, test_x, test_y = import_data(source, batch_size)

parse_data_train = tf.data.Dataset.zip((train_x, train_y))
parse_data_test = tf.data.Dataset.zip((test_x, test_y))
model1 = sr1_model((720, 1280, 3), (1080, 1920, 3))

perf = fit_data(model1, parse_data_train, parse_data_test, epch, batch_size)
visualizer(perf, epch)
