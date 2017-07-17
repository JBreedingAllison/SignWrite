"""
Define class of functions for dealing with video and frame image data.
"""
# !/usr/bin/env python
import csv
import glob
import operator
import os.path
import random
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from image_processor import process_image


class DataSet():
    def __init__(self, num_frames=60, class_limit=None,
                 image_shape=(224, 224, 3)):
        """
        Initialize
        params:
        num_frames = the number of frames to consider
        class_limit = limit for the number of classes (None = no limit)
        image_shape = the shape of the image
        """
        self.num_frames = num_frames
        self.class_limit = class_limit
        self.sequence_path = './sequences/'
        self.max_frames = 300  # set max number of frames to consider

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Clean that data!
        self.data = self.clean_data()

        # Get the image shape.
        self.image_shape = image_shape

    @staticmethod
    # Load data from extraction_data.csv file.
    def get_data():
        with open('./extraction_data.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    # Clean the data by limiting samples.
    def clean_data(self):
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.num_frames and int(item[3]) \
                    <= self.max_frames and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    # Get classes with respect to our limits.
    def get_classes(self):
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort the classes.
        classes = sorted(classes)

        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    # One-hot encode a class as a string.
    def get_class_one_hot(self, class_str):
        # Encode.
        label_encoded = self.classes.index(class_str)

        # One-hot encode.
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot = label_hot[0]

        return label_hot

    # Split data into train/test data.
    def split_train_test(self):
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, batch_Size, train_test, data_type,
                                    concat=False):
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Getting %s data with %d samples." % (train_test, len(data)))

        X, y = [], []
        for row in data:

            sequence = self.get_extracted_sequence(data_type, row)

            if sequence is None:
                print("No sequence found.")
                raise

            if concat:
                # Pass the sequence back as a single array.
                sequence = np.concatenate(sequence).ravel()

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples..." % (train_test,
                                                            len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset sequence to None.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                if data_type is "images":
                    # Get frames then rescale.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.num_frames)

                    # Build image sequence.
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the extracted sequence.
                    sequence = self.get_extracted_sequence(data_type, sample)

                if sequence is None:
                    print("Can't find sequence.")
                    sys.exit()

                if concat:
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        filename = sample[2]
        path = self.sequence_path + filename + '-' + str(self.num_frames) + \
            '-' + data_type + '.txt'
        if os.path.isfile(path):
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        path = './data/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        assert len(input_list) >= size

        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        return output[:size]

    @staticmethod
    def print_class_from_prediction(predictions, nb_to_return=5):
        # Get the prediction for each ASL sign.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[i]

        # Sort labels.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top k labels.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
