import sys
import os.path
import numpy as np
import h5py

from ..vicon_mocap import Body, Joint

# From Human3.6m dataset

def get_labels():
    return {"directions":1,
            "discussion":2,
            "eating":3,
            "greeting":4,
            "phoning":5,
            "photo":6,
            "posing":7,
            "purchases":8,
            "sitting":9,
            "sittingdown":10,
            "smoking":11,
            "waiting":12,
            "walkdog":13,
            "walking":14,
            "walktogether":15}

class BodyFileReader(object):
    '''
    Read and parse Human3.6m skeleton file.
    '''
    def __init__(self, file_path):
        '''
        Initialize BodyFileReader object.

        Args:
            file_path(str): path of the skeleton file.
        '''
        self._file_path = file_path
        self._frames = self._read(file_path)

    def __len__(self):
        '''
        Return the number of frames in the file.
        '''
        return len(self._frames)

    def __iter__(self):
        '''
        Iterate through each frame in the file, each frame can have more than one body.
        '''
        for frame in self._frames:
            yield frame

    def __getitem__(self, key):
        '''
        Index accessor to the loaded frames.

        Args:
            key(int): index of the body frame to return.
        '''
        return self._frames[key]

    def _read(self, path):
        '''
        Read Human3.6m skeleton dataset, the code is based on:
            https://github.com/una-dinosauria/3d-pose-baseline

        Args:
            path(str): file path to the skeleton file.
        '''
        frames = []
        if os.path.splitext(path)[1] == '.h5':
            with h5py.File(path, 'r') as h5f:
                poses = h5f['3D_positions'][:].T
                for i in range(poses.shape[0]):
                    body = Body()
                    joints = poses[i]
                    joint_count = joints.shape[0]//3
                    joints = np.reshape(joints, (joint_count, 3))
                    body.add_joints(joints)

                    body_frame = [body]
                    frames.append(body_frame)
        else:
            raise Exception('Unsupported file.')
        return frames
