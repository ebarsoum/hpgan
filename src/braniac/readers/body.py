import sys
import csv
import numpy as np

from braniac.format import SourceFactory

class SequenceBodyReader(object):
    '''
    Prepare a batch of sequence of bodies for each clip.
    '''
    def __init__(self, map_file, sequence_length, dataset, skip_frame=0, 
                 data_preprocessing=None, random_sequence=False, label_count=None, 
                 in_memory=True, camera_data_file=None, is_training=True,
                 seed=None):
        '''
        Initialize SequenceBodyReader which return a batch of sequences of
        Body packed in a numpy array.

        Args:
            map_file(str): path to the CSV file that contains the list of clips.
            sequence_length(int): the number of frames in the sequence.
            dataset(str): the name of the dataset.
            skip_frame(int): how many frames to skips.
            data_preprocessing(DataPreprocessing): responsible of normalizing the input data.
            random_sequence(bool): pick a random sequence from the clip.
            label_count(optional, int): assuming the label range from 0 to label_count-1, none
                                        mean no label provided.
            in_memory(bool): load the entire dataset in memory or not.
            camera_data_file(str): contains camera calibration data.
            is_training(bool): true mean shuffle the input.
            seed(int): seed used for the random number generator, can be None.
        '''
        self._source = SourceFactory(dataset, camera_data_file)
        self._map_file = map_file
        self._label_count = label_count
        self._sequence_length = sequence_length
        self._data_preprocessing = data_preprocessing
        self._files = []
        self._targets = []
        self._batch_start = 0
        self._skip_frame = skip_frame
        self._random_sequence = random_sequence
        self._in_memory = in_memory
        self._files.clear()
        self._sensor = self._source.create_sensor() 
        self._body = self._source.create_body()
        self._feature_shape = (self._body.joint_count, 3)

        assert self._skip_frame >= 0

        with open(map_file) as csv_file:
            data = csv.reader(csv_file)
            for row in data:
                # file path, activity id, subject id
                filename_or_object = self._source.create_file_reader(row[0]) \
                                     if self._in_memory else row[0]
                self._files.append([filename_or_object, int(row[1]), int(row[2])])
                if (self._label_count is not None) and (len(row) > 1):
                    target = [0.0] * self._label_count
                    target[int(row[1])] = 1.0
                    self._targets.append(target)

        self._indices = np.arange(len(self._files))
        if is_training:
            if seed != None:
                np.random.seed(seed)
            np.random.shuffle(self._indices)

    def size(self):
        return len(self._files)

    @property
    def element_shape(self):
        return self._feature_shape

    def has_more(self):
        if self._batch_start < self.size():
            return True
        return False

    def reset(self):
        self._batch_start = 0

    def next_minibatch(self, batch_size):
        '''
        Return a mini batch of sequences and their ground truth.

        Args:
            batch_size(int): mini batch size.
        '''
        batch_end = min(self._batch_start + batch_size, self.size())
        current_batch_size = batch_end - self._batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')

        inputs = np.empty(shape=(current_batch_size, self._sequence_length) + self._feature_shape, dtype=np.float32)
        activities = np.zeros(shape=(current_batch_size), dtype=np.int32)
        subjects = np.zeros(shape=(current_batch_size), dtype=np.int32)

        targets = None
        if self._label_count is not None:
            targets = np.empty(shape=(current_batch_size, self._label_count), dtype=np.float32)

        for idx in range(self._batch_start, batch_end):
            index = self._indices[idx]
            frames = self._files[index][0] if self._in_memory else self._source.create_file_reader(self._files[index][0])

            inputs[idx - self._batch_start, :, :, :] = self._select_frames(frames)
            activities[idx - self._batch_start] = self._files[index][1]
            subjects[idx - self._batch_start] = self._files[index][2]

            if self._label_count is not None:
                targets[idx - self._batch_start, :] = self._targets[index]

        self._batch_start += current_batch_size
        return inputs, targets, current_batch_size, activities, subjects

    def _select_frames(self, frames):
        '''
        Return a fixed sequence length from the provided clip.

        Args:
            file_path(str): path of the skeleton file to load.
        '''
        assert self._skip_frame >= 0
        num_frames = len(frames)
        multiplier = self._skip_frame + 1

        if not self._random_sequence:
            features = []
            if num_frames >= multiplier * self._sequence_length:
                start_frame = int(num_frames / 2 - (multiplier * self._sequence_length) / 2)
                for index in range(multiplier * self._sequence_length):
                    if (index % multiplier) == 0:
                        features.append(self._from_body_to_feature(frames[start_frame + index]))
            else:
                raise ValueError("Clip is too small, it has {} frames only.".format(num_frames))

            return np.stack(features, axis=0)
        else:
            features = []
            if num_frames >= multiplier * self._sequence_length:
                low = 0
                high = num_frames - multiplier * self._sequence_length + 1
                start = np.random.randint(low, high)
                for index in range(multiplier * self._sequence_length):
                    if (index % multiplier) == 0:
                        features.append(self._from_body_to_feature(frames[start + index]))
            else:
                raise ValueError("Clip is too small, it has {} frames only.".format(num_frames))

            return np.stack(features, axis=0)

    def _from_body_to_feature(self, frame):
        '''
        Convert body joints to a numpy array and apply the needed normalization.

        Args:
            frame: contain one or more body object.
        '''
        if len(frame) > 0:
            body = frame[0]
            return self._data_preprocessing.normalize(body.as_numpy())
        return None
