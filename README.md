# SignWrite: Transcribing Sign Language with the Power of Deep Learning

The Python files provided here will build an InceptionV3-based CNN to classify videos of a person signing in American Sign Language (ASL). The initial version was trained to classify signs of the 26 letters of the alphabet, but can be built to account for the complete dictionary of ASL.


## Requirements

keras == 2.0.2
numpy >= 1.12.1
pandas >= 0.19.2
tqdm >= 4.11.2
matplotlib >= 2.0.0
youtube-dl >= 2017.05.29


## Getting the data to build the model.

This version is designed to be trained on the letters of the alphabet. You can get the data you need to build the model from ASL tutorials on YouTube. If you have youtube-dl installed, you can download the videos in 'mp4' format by, for example, entering the following in a terminal window:

youtube-dl 39kghfhXtF4 -o abc7.mp4

All of the video data files must be placed in a './data' folder in your working directory.

For a more accurate model, you will want to collect many videos for each sign in your dictionary. In the case of the first version of this program, I collected videos that contained signs for each of the 26 letters of the alphabet. This set was split into training and testing videos. In the first version, 13 videos per letter were used for training and 5 videos per letter were used for testing. Below are the percentages of instances that the model scored the correct sign in the top k ranked signs:

Top  1:  5.61%
Top  3: 20.56%
Top 10: 59.81%


## Identifying start and end time for the signs in your dictionary.

Once you have gathered your data and placed it in the 'data' folder, you will then create a video_clip_times.csv file. This file will contain time information for your signs. That is, you will indicate the signs in your dictionary and when exactly they are shown in each of your video data.

For example, part of the video_clip_times.csv file will look like:

A, 20, 40, 16, ...
A, 21, 42, 17, ...
B, 27, 51, 18, ...
B, 29, 56, 20, ...
C, 35, 65, 20, ...
C, 38, 67, 22, ...

This format indicates that the sign for the letter 'A' was in the time interval (in seconds) [20,21] in abc1.mp4, in the time interval [40,42] in abc2.mp4, in the time interval [16,17] in abc3.mp4, etc. The sign for the letter 'B' was in the time interval [27,29] in abc1.mp4, in the time interval [51,56] in abc2.mp4, in the time interval [18,20] in abc3.mp4, etc.


## Sorting the video clips for each sign.

Now that the data has been collected and subclips identified, you then must extract the subclips and put them into the appropriate labelled folder in a train or test folder. This is done by running 

clip_videos_and_sort.py


## Extracting frames.

The next step is to extract jpgs of frames of all the subclips. This is done by running

extract_frame_jpgs.py

The data preparation is then complete.


## Building the model.

The InceptionV3 based convolutional neural network to classify signs can now be built. This is done by running

train_model.py

You must be connected to the internet to start building your model because it loads initial weights for the model from

https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


## TensorBoard

When building the model, log files are created for TensorBoard, a useful suite of visualization tools for understanding the training of the model. To access TensorBoard, run the following in a terminal window:

tensorboard --logdir=/THE_LOG_DIRECTORY

after replacing THE_LOG_DIRECTORY with the directory of the created log files.
