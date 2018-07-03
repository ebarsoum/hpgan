class SourceFactory:
    '''
    Create sensor and format instances for a specific dataset.
    '''
    def __init__(self, dataset, camera_data_file):
        '''
        Initialize SourceFactory with the dataset name and the calibration file.

        Args:
            dataset(str): the name of the dataset.
            camera_data_file(str): contains camera calibration data.
        '''
        self._dataset = dataset
        self._camera_data_file = camera_data_file

    def create_sensor(self):
        if self._dataset == "nturgbd":
            from ..format.kinect_v2 import Sensor
            return Sensor()
        elif self._dataset == "human36m":
            from ..format.vicon_mocap import Sensor
            return Sensor(self._camera_data_file)
    
    def create_body(self):
        if self._dataset == "nturgbd":
            from ..format.kinect_v2 import Body
            return Body(-1)
        elif self._dataset == "human36m":
            from ..format.vicon_mocap import Body
            return Body()

    def create_file_reader(self, file_path):
        if self._dataset == "nturgbd":
            from ..format.nturgbd.body import BodyFileReader
            return BodyFileReader(file_path)
        elif self._dataset == "human36m":
            from ..format.human36m.body import BodyFileReader
            return BodyFileReader(file_path)
