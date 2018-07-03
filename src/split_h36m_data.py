import os
import sys
import argparse
import csv
import numpy as np
import h5py

from braniac.utils import *
from braniac.format.human36m.body import BodyFileReader, get_labels

class DataSplitter(object):
    '''
    A helper class to split training data and generate a CSV files with the list of training data.
    '''
    def __init__(self, input_folder):
        '''
        Initialize DataSplitter object.

        Args:
            input_folder(str): path of the input folder that contains all the clips.
        '''
        self._input_folder = input_folder
        self._items = []
        self._stats_context = DataStatisticsContext()

    def load_data_paths(self):
        '''
        Load the list of files or sub-folders into a python list with their
        corresponding label.
        '''
        labels = get_labels()
        sub_folders = os.listdir(self._input_folder)
        index = 0
        self._items.clear()
        for folder in sub_folders:
            folder_path = os.path.join(self._input_folder, folder)
            if not os.path.isdir(folder_path):
                continue
            
            subject_id = int(folder[1:])
            files = os.listdir(folder_path)
            for item in files:
                item_path = os.path.join(folder_path, item)
                if not os.path.isfile(item_path):
                    continue
                
                # Get the filename without extension and remove anything after
                # the space, so `Directive 1` will become `Directive`.
                filename = os.path.splitext(item)[0].split()[0]
                if self._filter_data(item_path):
                    self._items.append([os.path.abspath(item_path), labels[filename.lower()], subject_id])

                if (index % 100) == 0:
                    print("Process {} items.".format(index+1))
                index += 1
        
        return self._items

    def _filter_data(self, item_path):
        '''
        Return True to add this item and false otherwise.

        Args:
            item_path(str): path of the item.

        Todo: Refactor filter.
        '''
        frames = BodyFileReader(item_path)
        if len(frames) >= 60:
            for frame in frames:
                if len(frame) == 0:
                    return False
            return True
        return False

    def split_data(self, items):
        '''
        Split the data at random for train, eval and test set.

        Args:
            items: list of clips and their correspodning label if available.
        '''
        item_count = len(items)
        indices = np.arange(item_count)
        np.random.shuffle(indices)

        train_count = int(0.8 * item_count)
        test_count  = item_count - train_count

        train = []
        test  = []

        for i in range(train_count):
            train.append(items[indices[i]])

        for i in range(train_count, train_count + test_count):
            test.append(items[indices[i]])

        return train, test

    def write_to_csv(self, items, file_path):
        '''
        Write file path and its target pair in a CSV file format.

        Args:
            items: list of paths and their corresponding label if provided.
            file_path(str): target file path.
        '''
        if sys.version_info[0] < 3:
            with open(file_path, 'wb') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for item in items:
                    writer.writerow(item)
        else:
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for item in items:
                    writer.writerow(item)

    def compute_statistics(self):
        '''
        Compute some statistics across all the datatset.
        '''
        with BodyDataStatisticsPass1(self._stats_context) as stats:
            for item in self._items:
                frames = BodyFileReader(item[0])
                for frame in frames:
                    stats.add(frame[0].as_numpy())

        with BodyDataStatisticsPass2(self._stats_context) as stats:
            for item in self._items:
                frames = BodyFileReader(item[0])
                for frame in frames:
                    stats.add(frame[0].as_numpy())

        return self._stats_context

def main(input_folder, output_folder):
    '''
    Main entry point, it iterates through all the clip files in a folder or through all
    sub-folders into a list with their corresponding target label. It then split the data
    into training set, validation set and test set.

    Args:
        input_folder: input folder contains all the data files.
        output_folder: where to store the result.
    '''
    data_splitter = DataSplitter(input_folder)
    items = data_splitter.load_data_paths()
    print("{} items loaded, start splitting.".format(len(items)))

    train, test = data_splitter.split_data(items)
    print("Train: {} and test: {}.".format(len(train), len(test)))

    context = data_splitter.compute_statistics()
    print("Complete computing statistics.")

    save_statistics_context(context, os.path.join(output_folder, 'data_statistics.h5'))

    if len(train) > 0:
        data_splitter.write_to_csv(train, os.path.join(output_folder, 'train_map.csv'))
    if len(test) > 0:
        data_splitter.write_to_csv(test, os.path.join(output_folder, 'test_map.csv'))

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_folder",
                        type = str,
                        help = "Input folder containing the raw data.",
                        required = True)

    parser.add_argument("-o",
                        "--output_folder",
                        type = str,
                        help = "Output folder for the generated training, validation and test text files.",
                        required = True)

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
    