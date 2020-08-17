from threading import Thread
import cv2
import numpy as np
import time


class VideoGet:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


def thread_both(source):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        video_shower.frame = frame


class IterateOverMovie:
    def __init__(self, params):
        self.stream = cv2.VideoCapture(params['Directory'])
        # (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.params = params
        self.video_details = {'Num Batches': 0,
                              'FPS': self.stream.get(cv2.CAP_PROP_FPS),
                              'Flag': 0,
                              'Num Frames': int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)),
                              'Batch Order': 0,
                              'Video': self.stream}
        self.array = 0

    def start(self):
        Thread(target=self.get_movie_details, args=()).start()
        print('Movie Details Received')
        return self

    def get(self, batch):
        Thread(target=self.run(batch), args=()).start()
        return self.array

    def get_movie_details(self):
        cap = self.stream
        debug = self.params['Debug']
        batch_size = self.params['Batch Size']
        num_frames = self.video_details['Num Frames']
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
        fps = self.video_details['FPS']
        if debug:
            print('Frame Rate: ', fps)
        batch_order = np.random.choice(num_batches, num_batches, replace=False)
        self.video_details = {'Num Batches': num_batches,
                              'FPS': fps,
                              'Flag': flag,
                              'Num Frames': num_frames,
                              'Batch Order': batch_order,
                              'Video': cap}
        return self.video_details

    def run(self, batch=0):
        num_batches = self.video_details['Num Batches']
        flag = self.video_details['Flag']
        num_frames = self.video_details['Num Frames']
        cap = self.video_details['Video']

        batch_size = self.params['Batch Size']
        init_shape = self.params['Init Shape']
        debug = self.params['Debug']

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
        self.array = array[1:, :, :, :]
        if debug:
            print(array.shape)


def lets_go(params1, params2):
    iterator720 = IterateOverMovie(params1).start()
    iterator1080 = IterateOverMovie(params2).start()
    for i in range(iterator720.video_details['Num Batches']):
        print('Batch : ' + str(i + 1) + ' out of ' + str(iterator720.video_details['Num Batches']) + '\r')
        batch = iterator720.video_details['Batch Order'][i]
        t0 = time.time()
        iterator720.get(batch)
        iterator1080.get(batch)
        t1 = time.time() - t0
        print('Time Taken : ', t1)
    print('Yay')


parameters1 = {'Directory': '../youtube/grownups720p.mkv',
               'Batch Size': 32,
               'Init Shape': (720, 1280, 3),
               'Destination': 'train/grownups_x/720p',
               'Seed': 534,
               'Debug': False}

parameters2 = {'Directory': '../youtube/grownups1080p.mkv',
               'Batch Size': 32,
               'Init Shape': (1080, 1920, 3),
               'Destination': 'train/grownups_x/720p',
               'Seed': 534,
               'Debug': False}

# lets_go(parameters1, parameters2)
