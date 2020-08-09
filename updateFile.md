# imageSR
A Live Image SR Model Using Very Deep RCNN


#### Update: 6th August 2020

A very basic version of the CNN is ready for testing. The model was trained using data from two main sources, DIV2K
data that you can find here and DAVIS 2017 dataset which you can find here. I came across a lot of problems in getting
the data-types right for input to the 'tf.keras.Model.fit' function, but managed to finally understand the problem and 
solve it. As it turns out, the problem was that the dataset object I was using was not labeled and thus was a dataset
of the shape (_batch_size_, 720, 1280, 3), whereas it should have been ((_batch_size_, 720, 1280, 3)__,__(_batch_size_,
1080, 1920, 3)), where the new addition is the output image which also is the output in this case.

It took more time than I realized but I found the bug and quashed it. The training was done for 10 epochs on a batch 
size of 16 over 7108 examples split 80:20 into training and validation (seed = 44). 
The architecture of my first attempt is as follows:

                         Height       Width      Channels
        Input Layer:      720    x    1280    x      3
        Rescaling:     (To bring pixel values between 0-1)
                          720    x    1280    x      3
        1st Conv:   (Filter=(5,5),  Strides=5, #Filters=128)
                          144    x     256    x     128
        2nd Conv:   (Filter=(4,4),  Strides=1, #Filters=150)
                          141    x     253    x     150
        3rd Conv:   (Filter=(7,14), Strides=1, #Filters=192)
                                (Padding = 'valid')
                          135    x     240    x     192
        Depth2Space:   (To reshape it into desired output)
                                (batch_size = 8)
                          1080   x     1920   x      3

Each convolution layer was followed by a 'ReLu' activation. The reason the convolution layers were non-symmetric is because of the last layer. With the target of 1080p in mind, and
the way Depth2Space works, I had to ensure the effective number of channels was divisible by (_batch_size_ ^2) and the
height of the image should be (_target_height_ / batch_size). These calculations helped me derive the rest of the layers.

This is okay for a first go, but it is most certainly not scalable, and I am going try to change that in the next iteration.
The total number of parameters for this current architecture was __3,139,670__. The next step is to increase the layers,
 and tweak the hyperparameters to increase the efficiency of the model. The metrics to gauge performance are as follows
  (in that order):
 
        Time Elapsed        PSNR        #Parameters
        
 The goal is to find the model  which takes the least amount of time to convert 30 low resolution frames to 30 high 
 resolution frames, while maintaining a very good Peak Signal to Noise Ratio, and also keeps the number of parameters 
 low leading to a lighter model capable of being deployed on smaller devices. As Andrew says, build the first one and
 keep iterating!
 
 
#### Update: 9th August 2020
Turns out data is indeed everything in this matter. I ran the architecture for 10 epochs and couldn't get past a local minima,
and that resulted in a very low PSNR of 12 dB. That means the image wasn't even visible properly. So two steps could be done
from here on, either build a deeper network, or get more data.

I built a bigger network without increasing the amount of data. I changed the model about 5 times and even ran them for 30 epochs
only to realize there was no effect.

I am still around 12 dB of PSNR and a non-viewable output image. This was when I decided to go through the research papers again 
to try and find if there was anything I missed. And without explicitly mentioning it one paper said it used a batch 
size of 64 and 9960 iterations over 80 epochs. And if my math is right, that means they roughly had a training size of about 600k images.
I was using 6k. That definitely could be one of the reasons why my network was failing me and it wasn't doing so for them.

I also read about the strategy they used, regarding residuals and creating a composite image to feed to the network. For those of you who 
want to read it, here's the link: https://arxiv.org/abs/1511.04587

My strategy now has shifted a bit since I started and now I'm currently focusing on getting the data ready for feeding it to this network.
My GPU is having quite the workout! I'll upload my specs and the architecture I'm using soon.