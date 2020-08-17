# imageSR
A Live Image SR Model Using Very Deep RCNN


## Update: 6th August 2020

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
 
 
## Update: 9th August 2020
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



## Update: 17th August 2020
First major update, and I'm really excited to present what I was able to make. This version now has a new custom model, a data handler, a test file (where I learnt how to use threading) and the main python file. It also has the model weights that match a performance of 29.5 dB PSNR, and I have not even completed my first epoch while training it.

### About the Model

The modified model uses multiple concepts and binds them together in what I call the U-Net (but can also be called a V-Net). As I was looking for unique models to use for this task, I came across four important concepts; _convolutional autoencoders_, _residual blocks_, _transposed convolution_ and _skip connections_. Convolutional autoencoders are special convolution networks that go through stages of 'encoding' and 'decoding' the output, more often than not 'recreating the input'. This is an important property that I felt needs to be a part of the model. A brief understanding of the concept of autoencoders can be understood by this:

![Image of Autoencoder] (https://miro.medium.com/max/3148/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png)

As the image shows, the output is a reconstructed version of the input after it goes through multiple stages. In the case of the convolutional autoencoder, the input goes through multiple stages of convolution.

The next important concept was the residual network. Residual CNNs are very famous and are well documented to allow us to go deeper (in terms of layers) without the problem of vanishing and exploding gradients. That is certainly helpful for this application. Transposed convolution is often misunderstood and misrepresented as 'deconvolution' and couldn't be further apart. Transposed convolution is the technique of passing a filter across an input image of smaller dimensions, to provide an output image of larger dimensions. This is a very useful technique and is something I used in the U-Net. To understand why skip connections are an important part of the model, first we need to see an important curve first (which also gives this network it's name).

                               ENCODER           |          DECODER              |          OUTPUT CONVERSION
                  2560x1440 =====================|==================== 2560x1440 ====>         2560x1440 
                                                 |                               |                 ||
                  1920x1080 =====================|====================           |                 ||
                                                 |                               |                 ||
                   1600x900 =====================|====================           |                 ||
                                                 |                               |                 ||
                   1366x768 =====================|====================           |               <====>
                                                 |                               |          target resolution
                   1280x720 =====================|==================== 1280x720  |             1920 x 1080
                        640x360 =================|================ 640x360       |             1600 x 900
                              320x180 ===========|========= 320x180              |             1366 x 768
                                    160x90 ======|===== 160x90                   |                          
                                        80x45   16:9   80x45                     |


As the diagram shows, the resolutions are mapped out in a rather simple manner, especially since the same resolutions are both, at the input and the output. To understand the model, let's use an example. Let's say your input is at 1280x720 and you want an output resolution of 1920x1080. In that case, the model will first convert 720p to 640x360, then to 320x180, then to 160x90 and 80x45. After some dense layers, and reducing 720p to a 16:9 dense & flat layer, the encoding process ends. For decoding, the concept of transposed convolution is used along with reshaping layers to ensure smooth rise up the decoding leg of the U-Net. From 16x9, the image is convolved with a filter with fractional stride (another valid explanation of transposed convolution) to reach 80x45, the same is done repeatedly to reach 160x90, 320x180, 640x360, 1280x720 and finally to 2560x1440. After arriving at this resolution, the image is ready for output. But more often than not we need more available resolutions like 1080p(HD) and limited by the capabilities of the screen. For this, the output is then 'downscaled to the desired resolution'. As downscaling is not a very intensive task, it produces desired quality at almost no performance cost.

There is however a critical aspect of skip connections, that makes this process much faster and also more reliable. Using bicubic upsampling was the way to upsample for many years before AI based methods were used. Here I decided to use skip connections between 'same resolutions' to improve performance. In the above example, there are skip connections between 1280x720 (i/o), 640x360 (i/o), 320x180 (i/o), 160x90 (i/o), and 80x45 (i/o). This provides significant performance gains. I also upsample the input using bicubic interpolation to the target resolution before adding it at the output stage. This brought an enormous boost to the accuracy and is a technique that has been used before. 

Basically irrespective of your input resolution, you first go down the encoder path, followed by going up the decoder path, and accordingly downsample the output to your desired resolution.

The whole model was made from scratch using Tensorflow, and if successful, I will also attempt to remake the model in PyTorch.

### About the Training Data
Deciding and finding what datasets to train this on, was a challenge. It was especially a hard challenge since I technically need 720p and 1080p versions of the same data. I then realised movies and video clips are a great source of this, and used multiple video clips of the same scene in both 720p (training x) and 1080p (ground truth - y). The next question was how to train the model on this data. I came up with essentially two strategies; one that takes a lot of space, and the other a lot of time. My first strategy was to first extract the frames from the movies and store them and then train the model on the images. My alternate strategy involved directly selecting random frames (in batches) from the video file and training the model with it.

While both were successful in training the model (yes, I tried them both), saving time is far more important than space, especially if you find out, you didn't program the right callback before it all began (and that happened too).

I decided I'll train the model 40k images at a time, out of a massive dataset of 600k that I managed to extract from couple of hours of diverse video clips. I trained the model on each super-batch of 40k images for 10 epochs, and because VRAM is so important, with a batch size of 2. I used a fairly low learning rate at 0.01, but still achieved those figures. For the loss function, I used the inbuilt Mean Squared Error function in Keras, and used the Adam optimizer. All the activations are 'ReLu', and I didn't see it fit to change it yet.

I also ensured (yes, much needed) to save the model weights every time I reached a new low for the loss.

### Next Steps
The next step is to keep training the data and see the performance after 10 epochs on each of the 600k images I have collected. It will be very useful to check the metrics at that point. But as it stands, 1.5 epochs down, test loss of 2.9467e-4 (which ~ equates to 35.31 dB, even higher than when I made the first commit of this update!) and a validation loss at the end of the epoch of 2.8748e-4 (which ~ equates to 35.4 dB). Will make my next update as I get closer to a much stronger result. 

#### Here's a Funny Joke/Tip
The one time I forgot to create the callback to save the weights for my model, I reached a test loss of 8.8223e-5. That means a PSNR of 40.54 dB. A value higher than state of the art. How about that. Don't be stupid. Save your model before it is too late.

