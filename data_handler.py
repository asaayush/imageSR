import cv2
import numpy as np
import tensorflow as tf
from threading import Thread
import time

# Example Directory
# 'youtube/video720p.mp4'


class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get(), args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


def threaded_get_video_det(directory):
    cap = VideoGet(directory).start()

    print('Thread Started')
    return cap, cap.stream


def get_video_det(params, debug=False):
    directory = params['directory']
    batch_size = params['batch_size']

    cap = cv2.VideoCapture(directory)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if debug:
        print('Number of Frames: ', num_frames)
    if num_frames % batch_size == 0:
        num_batches = int(num_frames / batch_size)
        flag = 0
    else:
        num_batches = int(num_frames / batch_size) + 1
        flag = 1
    if debug:
        print('Number of Batches: ', num_batches)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if debug:
        print('Frame Rate: ', fps)
    batch_order = np.random.choice(num_batches, num_batches, replace=False)
    # cap.release()
    # t0 = time.time()
    # ideo_get, cap = threaded_get_video_det(directory)
    # t1 = time.time() - t0
    # print('Thread Initializing took '+str(round(t1, 2))+' seconds.')
    video_details = {'Num Batches': num_batches,
                     'FPS': fps,
                     'Flag': flag,
                     'Num Frames': num_frames,
                     'Batch Order': batch_order,
                     'Video': cap},
#                     'ThreadedObj': video_get}
    return video_details


def import_data(batch, params, vid_details, cap=0, debug=False):
    num_batches = vid_details['Num Batches']
    flag = vid_details['Flag']
    num_frames = vid_details['Num Frames']
    cap = vid_details['Video']

    batch_size = params['batch_size']
    init_shape = params['init_shape']

    array = np.empty(shape=init_shape)
    array = np.expand_dims(array, axis=0)

    batch_sz = batch_size
    if (batch == num_batches) & (flag == 1):
        batch_sz = num_frames - ((num_batches - 1) * batch_size)
    while cap.isOpened():
        for j in range(batch_sz):
            if flag == 1:
                if debug:
                    print('Cursor Position : ', (batch - 1) * batch_size + j)
                cap.set(propId=cv2.CAP_PROP_POS_FRAMES, value=((batch - 1) * batch_size) + j)
            else:
                if debug:
                    print('Cursor Position : ', batch * batch_size + j)
                cap.set(propId=cv2.CAP_PROP_POS_FRAMES, value=(batch * batch_size) + j)
            if debug:
                print('Frame Number : ', j)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, dsize=(init_shape[1], init_shape[0]))
                frame = np.expand_dims(frame, axis=0)
                array = np.append(array, frame, axis=0)
        break
    array = array[1:, :, :, :]
    if debug:
        print(array.shape)
    return array


def store_data(params):
    cap = cv2.VideoCapture(params['directory'])
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame, dsize=(params['init_shape'][1], params['init_shape'][0]))
            cv2.imwrite(filename=params['destination']+'/'+str(count)+'.jpg', img=frame)
            count += 1
            if count % 100 == 0:
                print(str(count)+'/'+str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))+' images completed.')
    print('Done')


def get_data(params, size):
    train_x = tf.keras.preprocessing.image_dataset_from_directory(directory=params['Destination'],
                                                                  batch_size=params['Batch Size'],
                                                                  shuffle=True, seed=params['Seed'],
                                                                  validation_split=0.2,
                                                                  subset='training',
                                                                  image_size=size,
                                                                  label_mode=None)
    test_x = tf.keras.preprocessing.image_dataset_from_directory(directory=params['Destination'],
                                                                 batch_size=params['Batch Size'],
                                                                 shuffle=True, seed=params['Seed'],
                                                                 validation_split=0.2,
                                                                 subset='validation',
                                                                 image_size=size,
                                                                 label_mode=None)
    return train_x, test_x
