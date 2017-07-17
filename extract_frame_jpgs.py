"""
Extract frames in jpg format from clips of videos in test and train folders.
Make a csv file containing the number of frames extracted for each video.
"""
# !/usr/bin/env python
import csv
import glob
import os
import os.path
from subprocess import call


def frames_to_jpgs():
    """
    Extract frames of videos in '.avi, .flv, .mov, .mp4' format that are
    contained in subfolders of the train and test folders.
    """
    # Define list of folders to find videos to extract jpgs and initialize
    # a list to hold the extraction data.
    test_train_folders = ['./train/', './test/']
    extraction_data = []

    # Locate and process all videos in '.avi, .flv, .mov, .mp4' format
    # in the list of folders.
    for folder in test_train_folders:
        labeled_folders = glob.glob(folder + '*')

        for labeled_folder in labeled_folders:
            # Get all video files in '.avi, .flv, .mov, .mp4' format
            avi_files = glob.glob(labeled_folder + '/*.avi')
            flv_files = glob.glob(labeled_folder + '/*.flv')
            mov_files = glob.glob(labeled_folder + '/*.mov')
            mp4_files = glob.glob(labeled_folder + '/*.mp4')

            # Put all video files in one big list.
            video_files = []
            video_files.extend(avi_files + flv_files + mov_files + mp4_files)

            for video_file in video_files:
                # Get useful pieces of each video's filename.
                video_filename_parts = filename_parts_extractor(video_file)

                train_test_label, folder_name, filename_no_ext, filename = video_filename_parts

                # Extract frames if not already done.
                if not is_extracted(video_filename_parts):
                    src = train_test_label + '/' + folder_name + '/' + filename
                    dest = train_test_label + '/' + folder_name + '/' + \
                        filename_no_ext + '-%04d.jpg'
                    call(["ffmpeg", "-i", src, dest])

                # Count the number of frames.
                frame_count = count_frames(video_filename_parts)

                extraction_data.append([train_test_label, folder_name,
                                        filename_no_ext, frame_count])

                print("Generated %d frames for %s" % (frame_count,
                                                      filename_no_ext))

    with open('extraction_data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(extraction_data)

    print("Extracted frames for %d video files." % (len(extraction_data)))


def count_frames(video_filename_parts):
    '''
    Counts the number of frames extracted for a video in a given path.
    params:
    video_filename_parts = train_or test, folder_name, \
                           filename_no_ext, filename
    '''
    train_test_label, folder_name, filename_no_ext, _ = video_filename_parts
    frame_jpg_count = glob.glob(train_test_label + '/' + folder_name + '/' +
                                filename_no_ext + '*.jpg')
    return len(frame_jpg_count)


def filename_parts_extractor(video_file):
    '''
    Returns the different parts of a filename's name and path.
    params:
    video_file = the complete path of the file.
    output:
    test/train label, folder name, filename without extension, filename
    '''
    parts = video_file.split('/')
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    folder_name = parts[2]
    train_test_label = parts[1]

    return train_test_label, folder_name, filename_no_ext, filename


def is_extracted(video_filename_parts):
    '''
    Determine if jps of frames have been extracted from a video.
    params:
    name_parts = test/train label, class name, filename without extension,
                filename
    output:
    True/False
    '''
    train_test_label, folder_name, filename_no_ext, _ = video_filename_parts
    return bool(os.path.exists(train_test_label + '/' + folder_name +
                               '/' + filename_no_ext + '-0001.jpg'))


def main():
    frames_to_jpgs()


if __name__ == '__main__':
    main()
