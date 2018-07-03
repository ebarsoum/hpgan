import numpy as np
from enum import Enum
import h5py

class DataStatisticsContext:
    '''
    A context class hold data statistics values.
    '''
    def __init__(self):
        self.clear()

    def clear(self):
        self.count = 0
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.global_mean = None
        self.global_std = None

class NormalizationMode(Enum):
    ''' The different modes in normalizing the input data '''
    MeanAndStd = 0, # Remove mean and divide by standard deviation.
    MeanAndStd2 = 1, # Remove mean and divide by double standard deviation.
    GlobalMeanAndStd = 2, # Remove global mean and divide by standard deviation.
    GlobalMeanAndStd2 = 3, # Remove global mean and divide by double standard deviation.
    MinAndMax = 4, # Map min and max to -1 and 1.
    External = 5 # Use the provided class. 

def save_statistics_context(context, file_name):
    with h5py.File(file_name, 'w') as h5f:
        h5f.create_dataset('mean', data=context.mean)
        h5f.create_dataset('std', data=context.std)
        h5f.create_dataset('count', data=(context.count,))
        h5f.create_dataset('min', data=context.min)
        h5f.create_dataset('max', data=context.max)
        h5f.create_dataset('global_mean', data=context.global_mean)
        h5f.create_dataset('global_std', data=context.global_std)

def load_statistics_context(file_name):
    context = DataStatisticsContext()
    with h5py.File(file_name,'r') as h5f:
        context.mean = h5f['mean'][:]
        context.std = h5f['std'][:]
        context.count = h5f['count'][0]
        context.min = h5f['min'][:]
        context.max = h5f['max'][:]
        context.global_mean = h5f['global_mean'][:]
        context.global_std = h5f['global_std'][:]

    return context

class DataPreprocessing:
    '''
    A helper class that normalize raw data based on the provide statistics.
    '''
    def __init__(self, 
                 data_statisitcs_file,
                 normalization_mode=NormalizationMode.MeanAndStd,
                 normalizer=None,
                 remove_center_of_gravity=False):
        self._normalization_mode = normalization_mode
        self._stats_data = load_statistics_context(data_statisitcs_file)
        self._normalizer = normalizer
        self._remove_center_of_gravity = remove_center_of_gravity
    
    def normalize(self, inputs):
        norm = None
        if self._normalization_mode == NormalizationMode.MeanAndStd:
            norm = self._mean_std_normalize(inputs)
        elif self._normalization_mode == NormalizationMode.MeanAndStd2:
            norm = self._mean_std_normalize(inputs, 2.0)
        elif self._normalization_mode == NormalizationMode.GlobalMeanAndStd:
            norm = self._global_mean_std_normalize(inputs)
        elif self._normalization_mode == NormalizationMode.GlobalMeanAndStd2:
            norm = self._global_mean_std_normalize(inputs, 2.0)
        elif self._normalization_mode == NormalizationMode.MinAndMax:
            norm = self._min_max_normalize(inputs)
        elif self._normalization_mode == NormalizationMode.External:
            if self._normalizer == None:
                raise Exception('Normalizer is missing.')
            norm = self._normalizer.normalize(inputs)

        if self._remove_center_of_gravity:
            norm = self.remove_center_of_gravity(norm)
        return norm

    def unnormalize(self, inputs):
        unnorm = None
        if self._normalization_mode == NormalizationMode.MeanAndStd:
            unnorm = self._mean_std_unnormalize(inputs)
        elif self._normalization_mode == NormalizationMode.MeanAndStd2:
            unnorm = self._mean_std_unnormalize(inputs, 2.0)
        elif self._normalization_mode == NormalizationMode.GlobalMeanAndStd:
            unnorm = self._global_mean_std_unnormalize(inputs)
        elif self._normalization_mode == NormalizationMode.GlobalMeanAndStd2:
            unnorm = self._global_mean_std_unnormalize(inputs, 2.0)
        elif self._normalization_mode == NormalizationMode.MinAndMax:
            unnorm = self._min_max_unnormalize(inputs)
        elif self._normalization_mode == NormalizationMode.External:
            if self._normalizer == None:
                raise Exception('Normalizer is missing.')
            unnorm = self._normalizer.unnormalize(inputs)
        
        return unnorm

    def _mean_std_normalize(self, inputs, std_factor=1.0):
        return (inputs - self._stats_data.mean) / (self._stats_data.std * std_factor)

    def _mean_std_unnormalize(self, inputs, std_factor=1.0):
        return (inputs * self._stats_data.std * std_factor) + self._stats_data.mean

    def _global_mean_std_normalize(self, inputs, std_factor=1.0):
        return (inputs - self._stats_data.global_mean) / (self._stats_data.global_std * std_factor)

    def _global_mean_std_unnormalize(self, inputs, std_factor=1.0):
        return (inputs * self._stats_data.global_std * std_factor) + self._stats_data.global_mean

    def _min_max_normalize(self, inputs):
        max_ = np.amax(self._stats_data.max, axis=0)
        min_ = np.amin(self._stats_data.min, axis=0)

        return 2.0 * ((inputs - min_) / (max_ - min_)) - 1.0
 
    def _min_max_unnormalize(self, inputs):
        max_ = np.amax(self._stats_data.max, axis=0)
        min_ = np.amin(self._stats_data.min, axis=0)

        return (max_ - min_) * ((inputs + 1.0) / 2.0) + min_

    def remove_center_of_gravity(self, inputs):
        return inputs - np.mean(inputs, axis=-2)

class BodyDataStatisticsPass1:
    '''
    A helper class that compute the mean and global mean for skeleton data.
    '''
    def __init__(self, context):
        self._context = context

    def __enter__(self):
        self._context.clear()
        return self

    def add(self, item):
        if self._context.mean is None:
            self._context.mean = item
            self._context.min = item
            self._context.max = item
        else:
            self._context.mean += item
            self._context.min = np.minimum(self._context.min, item)
            self._context.max = np.maximum(self._context.max, item)
            
        self._context.count += 1.0

    def __exit__(self, exc_type, exc_value, traceback):
        self._context.global_mean = np.sum(self._context.mean, axis=0)
        self._context.mean = self._context.mean / self._context.count
        self._context.global_mean = self._context.global_mean / (self._context.count * self._context.mean.shape[0])

class BodyDataStatisticsPass2:
    '''
    A helper class that compute the standard deviation and global std for skeleton data.
    '''
    def __init__(self, context):
        self._context = context        
        if (self._context.mean is None) or (self._context.global_mean is None):
            raise ValueError("Pass 1 need to run first.")

        self._context = context

    def __enter__(self):
        return self

    def add(self, item):
        if self._context.std is None:
            self._context.std = (item - self._context.mean)**2
            self._context.global_std = np.sum((item - self._context.global_mean)**2, axis=0)
        else:
            self._context.std += (item - self._context.mean)**2
            self._context.global_std += np.sum((item - self._context.global_mean)**2, axis=0)
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._context.std = np.sqrt(self._context.std / self._context.count)
        self._context.global_std = np.sqrt(self._context.global_std / (self._context.count * self._context.mean.shape[0]))
