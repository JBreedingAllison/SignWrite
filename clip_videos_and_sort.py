'''
Extracts clips of videos in the data folder with start and end times designated
by video_clip_times.csv and sorts them into train and test folders
'''
# !/usr/bin/env python
import csv
import os
import fnmatch
import moviepy.editor as mpy

# Count the number of videos in mp4 format in the data folder.
cwd = os.getcwd()
data_dir = cwd + '/data'
data_set = fnmatch.filter(os.listdir(data_dir), '*.mp4')
data_count = len(data_set)

# Identify the csv file with subclip start and end times.
data_csv = 'video_clip_times.csv'

# Create the list of signs, which consists solely of the alphabet.
signs = map(chr, range(65, 91))
signs_count = len(signs)

# Create folders for each of the signs in train and test folders.
for sign in signs:
    if not os.path.exists(cwd + '/train/' + sign):
        os.makedirs(cwd + '/train/' + sign)
    if not os.path.exists(cwd + '/test/' + sign):
        os.makedirs(cwd + '/test/' + sign)


# Create lists to store subclip start and end times for each video.
start_times = []
end_times = []
for i in range(0, data_count):
    start_times.append([])
    end_times.append([])

# Read the start and end times for each clip
with open(data_csv) as csvfile:
    read_csv = csv.reader(csvfile, delimiter=',')

    row_count = 0
    for row in read_csv:
        if row_count % 2 == 0:
            for video_index in range(0, data_count):
                start_times[video_index].append(float(row[video_index+1]))
        if row_count % 2 == 1:
            for video_index in range(0, data_count):
                end_times[video_index].append(float(row[video_index+1]))
        row_count += 1

    # Split the data into train/test sets following the 80:20 ratio.
    train_video_count = int(data_count*.8)
    test_video_count = data_count - train_video_count

    # Extract clips for the training videos and sort into a labelled sub folder
    # in the train or test folders.
    train_set = []
    test_set = []
    for video_index in range(0, data_count):
        # Identify if the video is in the test set or the train set and assign
        # it the corresponding label.
        if video_index <= train_video_count:
            train_test_label = 'train'
        else:
            train_test_label = 'test'

        # Get the video, resize and set the naming style.
        cur_video = mpy.VideoFileClip(data_dir + '/' + data_set[video_index])
        cur_duration = cur_video.duration
        cur_name_style = '_' + data_set[video_index]

        # Get the start and end times for all subclips.
        cur_start_times = start_times[video_index]
        cur_end_times = end_times[video_index]

        # Extract all subclips and put into the appropriate folder.
        for sign_index in range(0, signs_count):
            cur_dir = train_test_label + '/' + signs[sign_index] + '/'

            cur_clip_name = signs[sign_index] + cur_name_style
            cur_clip_start = cur_start_times[sign_index]
            cur_clip_end = cur_end_times[sign_index]

            cur_clip = cur_video.subclip(cur_clip_start, cur_clip_end)

            cur_clip.write_videofile(cur_dir + cur_clip_name, audio=False,
                                     fps=15)
