import tensorflow as tf
from keras import layers


def import_high_res(train_dir, batch, img_size, info=0):
    data_source_training = train_dir
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_training,
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=44,
                                                                   image_size=img_size,
                                                                   label_mode=None,
                                                                   batch_size=batch)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=data_source_training,
                                                                  validation_split=0.2,
                                                                  subset='validation',
                                                                  seed=44,
                                                                  image_size=img_size,
                                                                  label_mode=None,
                                                                  batch_size=batch)
    if info == 1:
        print("Training Data-set Samples   ~= " + str(len(train_ds)*batch))
        print("Training Data-set Length     = " + str(len(train_ds)))
        print("Training Data-set Type       = " + str(type(train_ds)))
        print("Validation Data-set Samples ~= " + str(len(test_ds) * batch))
        print("Validation Data-set Length   = " + str(len(test_ds)))
        print("Validation Data-set Type     = " + str(type(test_ds)))
    return train_ds, test_ds


def down_sizing(data_set, resize, string):
    count = 0
    print("Down-sizing the Data")
    for element in data_set:
        k = tf.squeeze(element)
        path = 'datasets/single_combined/hr/' + string + '/' + str(count) + '.png'
        tf.keras.preprocessing.image.save_img(path, k, data_format="channels_last", scale=True)
        k = tf.image.resize(k, resize, method='gaussian').numpy()
        path = 'datasets/single_combined/lr/' + string + '/' + str(count) + '.png'
        tf.keras.preprocessing.image.save_img(path, k, data_format="channels_last", scale=True)
        if count % 100 == 0:
            print("..." + str(count) + " samples complete...")
        count += 1
    print('Done')


def converting_2_low_res():
    batch_size = 1
    train_directory = 'datasets/single_combined/training'
    image_size = (1080, 1980)
    down_sampled_shape = (720, 1280)
    training_ds, testing_ds = import_high_res(train_directory, batch_size, image_size, info=1)
    down_sizing(training_ds, down_sampled_shape, 'training')
    down_sizing(testing_ds, down_sampled_shape, 'validation')


def espcn_model(image_shape):
    x_input = tf.keras.Input(image_shape)
    print(x_input)
    x = layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=image_shape)(x_input)
    x = layers.Conv2D(128, 5, activation='relu', strides=5, name='conv1')(x)
    x = layers.Conv2D(150, 4, activation='relu', name='conv2')(x)
    x = layers.Conv2D(192, kernel_size=(14, 7), activation='relu', padding='valid')(x)
    x = tf.nn.depth_to_space(input=x, block_size=8, data_format='NHWC')
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])
    model.summary()
    return model


converting_2_low_res()


# model1 = espcn_model((1280, 720, 3))
