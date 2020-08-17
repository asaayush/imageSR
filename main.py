# This is the main script

from UNetModel import *
from data_handler import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import cv2
import threading
import time
from trying_threading import IterateOverMovie

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print("Initializing Mixed Float Policy...")


parameters1 = {'Directory': 'youtube/grownups720p.mkv',
               'Batch Size': 2,
               'Init Shape': (720, 1280, 3),
               'Destination': 'train/x',
               'Seed': 534,
               'Debug': False}
parameters2 = {'Directory': 'youtube/grownups1080p.mkv',
               'Batch Size': 2,
               'Init Shape': (1080, 1920, 3),
               'Destination': 'train/y',
               'Seed': 534,
               'Debug': False}


def custom_method():
    # cap720 = threaded_get_video_det(parameters1)
    thread1 = threading.Thread(target=get_video_det, args=(parameters1, True))
    thread1.start()
    print('thread has started')

    vid_det_720 = get_video_det(parameters1, debug=True)

    # cap1080 = threaded_get_video_det(parameters2)
    vid_det_1080 = get_video_det(parameters2, debug=True)
    print("Both threads are ready")
    num_epochs = 1

    # sr_model = u_net_model((720, 1280, 3), (1080, 1920), 0.01, 3)
    sr_model = u_net_model((720, 1280, 3), (1080, 1920), 0.01, 4)
    t0 = time.time()
    for epoch in range(num_epochs):
        print('Epoch Number : ', (epoch+1))
        for i in range(vid_det_720['Num Batches']):
            print('Batch : ' + str(i + 1) + ' out of ' + str(vid_det_720['Num Batches']) + '\r')
            batch = vid_det_720['Batch Order'][i]

            # For Training 'x' at 720p
            train_x = import_data(batch, parameters1, vid_det_720)
            # For Training 'y' at 1080p
            train_y = import_data(batch, parameters2, vid_det_1080)
            train_y = train_y/255

            sr_model.fit(x=train_x, y=train_y, batch_size=1, validation_split=0.2)

            # Evaluating every 10 batches
            if (i+1) % 10 == 0:
                t3 = time.time() - t0
                print('Total Time Elapsed : ' + str(round(t3, 2)) + 'seconds    \r')
                kx = cv2.imread('images/test_image.jpg')
                ky = cv2.imread('images/test_image_y.jpg')
                kx = cv2.cvtColor(kx, cv2.COLOR_BGR2RGB)
                ky = cv2.cvtColor(ky, cv2.COLOR_BGR2RGB)
                ky = cv2.resize(ky, dsize=(1920, 1080))
                kx = np.expand_dims(kx, axis=0)
                ky = (np.expand_dims(ky, axis=0))/255
                sr_model.evaluate(x=kx, y=ky)
                t4 = time.time()
                k = sr_model(kx)

                t5 = time.time() - t4
                k = tf.squeeze(k)

                tf.keras.preprocessing.image.save_img(path='images/output_image.png', x=k)

                print('Time Taken for passing one image through network = ', t5)
            # Saving Weights every 50 batches
            if (i+1) % 50 == 0:
                sr_model.save_weights(filepath='models/weights'+str(i))

    # vid_det_720['Video'].release
    # vid_det_1080['Video'].release


def normal_method(params1, params2, create_data=False, load_weights=True):
    def pre_process(input_ds):
        return input_ds.map(lambda x: x / 255)

    # Choose Model
    sr_model = u_net_model((720, 1280, 3), (1080, 1920), 0.01, 4)

    if load_weights:
        sr_model.load_weights(filepath='model/checkpoints')

    if create_data:
        # Store 720p images
        store_data(params1)
        # Store 1080p images
        store_data(params2)

    # Source Data
    train_x, val_x = get_data(params1, (720, 1280))
    train_y, val_y = get_data(params2, (1080, 1920))

    # Prepare data for training
    train_y = train_y.apply(transformation_func=pre_process)
    val_y = val_y.apply(transformation_func=pre_process)

    train_ds = tf.data.Dataset.zip((train_x, train_y))
    valid_ds = tf.data.Dataset.zip((val_x, val_y))

    # Define CallBacks ==> Save Weights Checkpoint
    callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/checkpoints', monitor='val_loss', verbose=1,
                                                  save_freq=100, save_weights_only=True)

    # Fit the data
    sr_model.fit(x=train_ds, validation_data=valid_ds, epochs=10, batch_size=parameters1['Batch Size'],
                 callbacks=[callback])


def custom_with_threading(params1, params2):
    iterator720 = IterateOverMovie(params1).start()
    iterator1080 = IterateOverMovie(params2).start()
    sr_model = u_net_model((720, 1280, 3), (1080, 1920), 0.01, 3)
    for i in range(iterator720.video_details['Num Batches']):
        print('Batch : ' + str(i + 1) + ' out of ' + str(iterator720.video_details['Num Batches']) + '\r')
        batch = iterator720.video_details['Batch Order'][i]
        t0 = time.time()
        iterator720.get(batch)
        x = iterator720.array
        t1 = time.time() - t0
        iterator1080.get(batch)
        y = iterator1080.array/255
        t2 = time.time() - t0
        print('Time Taken : ', round(t2, 2), round(t1, 2))
        sr_model.fit(x=x, y=y, epochs=1, batch_size=1, validation_split=0.2)

        if (i + 1) % 10 == 0:
            t3 = time.time() - t0
            print('Total Time Elapsed : ' + str(round(t3, 2)) + 'seconds    \r')
            kx = cv2.imread('images/test_image.jpg')
            ky = cv2.imread('images/test_image_y.jpg')
            kx = cv2.cvtColor(kx, cv2.COLOR_BGR2RGB)
            ky = cv2.cvtColor(ky, cv2.COLOR_BGR2RGB)
            ky = cv2.resize(ky, dsize=(1920, 1080))
            kx = np.expand_dims(kx, axis=0)
            ky = (np.expand_dims(ky, axis=0)) / 255
            sr_model.evaluate(x=kx, y=ky)
            t4 = time.time()
            k = sr_model(kx)

            t5 = time.time() - t4
            k = tf.squeeze(k)

            tf.keras.preprocessing.image.save_img(path='images/output_image.png', x=k)

            print('Time Taken for passing one image through network = ', t5)
        # Saving Weights every 50 batches
        if (i + 1) % 50 == 0:
            sr_model.save_weights(filepath='models/weights' + str(i))


normal_method(parameters1, parameters2)

# custom_method()

# custom_with_threading(parameters1, parameters2)
