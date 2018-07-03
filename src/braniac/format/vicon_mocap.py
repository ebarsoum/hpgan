import sys
import math
import numpy as np
from enum import Enum
import h5py

#
# This MoCap is used for Human36m dataset:
#   http://vision.imar.ro/human3.6m/description.php
#

class Sensor(object):
    ''' Sensor object contains all sensor specific properties '''

    def __init__(self, calibration_file_path=None):
        self._width = 2*512.
        self._height = 2*424.

        self._cameras = None
        if calibration_file_path != None:
            self._cameras = H36MCameras(calibration_file_path)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def project_3d_to_2d(self, pose3d, subject_id, camera_index=0):
        return self._cameras.project_3d_to_2d(pose3d, subject_id, camera_index)

    def map_world_to_depth(self, x, y, z):
        ''' Map from word coordinate to 2D depth coordinate, based on OpenNI '''
        depth_x = self._width * (x/2400.) + self._width/2
        depth_y = self._height/2 - self._height * (y/2400.)

        return depth_x, depth_y

class H36MCameras(object):
    '''
    Camera intrinsic properties per subject per camera from:
        https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/cameras.py 
    '''
    def __init__(self, file_path):
        '''
        Load camera properties from h5 file.
        '''
        self._camera_count = 4
        self._cams = self._load_cameras(file_path)

    def project_3d_to_2d(self, pose3d, subject_id, camera_index):
        """
        Project 3d pose using camera parameters into 2d pose

        Args
            pose3d(nparray): input 3d pose to project.
            subject_id(int): subject id.
            camera_index(int): which camera to project to.

        Returns
            pose2d: project 2d pose for the specified camera.
        """
        assert (camera_index >= 0) and (camera_index < self._camera_count)

        R, T, f, c, k, p, name = self._cams[(subject_id, camera_index+1)]
        pose2d, _, _, _, _ = self._project_point_radial(pose3d, R, T, f, c, k, p)
        return np.reshape(pose2d, [-1, 2])

    def _project_point_radial(self, P, R, T, f, c, k, p):
        """
        Project points from 3d to 2d using camera parameters
        including radial and tangential distortion
        Args
            P: Nx3 points in world coordinates
            R: 3x3 Camera rotation matrix
            T: 3x1 Camera translation parameters
            f: (scalar) Camera focal length
            c: 2x1 Camera center
            k: 3x1 Camera radial distortion coefficients
            p: 2x1 Camera tangential distortion coefficients
        Returns
            Proj: Nx2 points in pixel space
            D: 1xN depth of each point in camera space
            radial: 1xN radial distortion per point
            tan: 1xN tangential distortion per point
            r2: 1xN squared radius of the projected points before distortion
        """
        # P is a matrix of 3-dimensional points
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        N = P.shape[0]
        X = R.dot( P.T - T ) # rotate and translate
        XX = X[:2,:] / X[2,:]
        r2 = XX[0,:]**2 + XX[1,:]**2

        radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) )
        tan = p[0]*XX[1,:] + p[1]*XX[0,:]

        XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

        Proj = (f * XXX) + c
        Proj = Proj.T

        D = X[2,]

        return Proj, D, radial, tan, r2

    def _world_to_camera_frame(self, P, R, T):
        """
        Convert points from world to camera coordinates
        Args
            P: Nx3 3d points in world coordinates
            R: 3x3 Camera rotation matrix
            T: 3x1 Camera translation parameters
        Returns
            X_cam: Nx3 3d points in camera coordinates
        """

        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = R.dot( P.T - T ) # rotate and translate

        return X_cam.T

    def _camera_to_world_frame(self, P, R, T):
        """
        Inverse of world_to_camera_frame
        Args
            P: Nx3 points in camera coordinates
            R: 3x3 Camera rotation matrix
            T: 3x1 Camera translation parameters
        Returns
            X_cam: Nx3 points in world coordinates
        """
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = R.T.dot( P.T ) + T # rotate and translate

        return X_cam.T

    def _load_camera_params(self, hf, path):
        """
        Load h36m camera parameters
        Args
            hf: hdf5 open file with h36m cameras data
            path: path or key inside hf to the camera we are interested in
        Returns
            R: 3x3 Camera rotation matrix
            T: 3x1 Camera translation parameters
            f: (scalar) Camera focal length
            c: 2x1 Camera center
            k: 3x1 Camera radial distortion coefficients
            p: 2x1 Camera tangential distortion coefficients
            name: String with camera id
        """
        R = hf[ path.format('R') ][:]
        R = R.T

        T = hf[ path.format('T') ][:]
        f = hf[ path.format('f') ][:]
        c = hf[ path.format('c') ][:]
        k = hf[ path.format('k') ][:]
        p = hf[ path.format('p') ][:]

        name = hf[ path.format('Name') ][:]
        name = "".join([chr(item) for item in name])

        return R, T, f, c, k, p, name

    def _load_cameras(self, file_path, subjects=[1,5,6,7,8,9,11]):
        """
        Loads the cameras of h36m
        Args
            file_path: path to hdf5 file with h36m camera data
            subjects: List of ints representing the subject IDs for which cameras are requested
        Returns
            rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
        """
        rcams = {}

        with h5py.File(file_path,'r') as hf:
            for s in subjects:
                for c in range(self._camera_count):
                    rcams[(s, c+1)] = self._load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1))

        return rcams

class JointType(Enum):
    ''' List of the joint types and their corresponding index '''
    Unknown = -1       # The type isn't available
    Hip = 0
    HipRight = 1
    KneeRight = 2
    FootRight = 3
    ToeBaseRight = 4
    Site1 = 5
    HipLeft = 6
    KneeLeft = 7
    FootLeft = 8
    ToeBaseLeft = 9
    Site2 = 10
    Spine1 = 11
    Spine2 = 12
    Neck = 13
    Head = 14
    Site3 = 15
    ShoulderLeft = 16
    ElbowLeft = 17
    WristLeft = 18
    HandLeft = 19
    HandThumbLeft = 20
    Site4 = 21
    WristEndLeft = 22
    Site5 = 23    
    ShoulderRight = 24
    ElbowRight = 25
    WristRight = 26
    HandRight = 27
    HandThumbRight = 28
    Site6 = 29
    WristEndRight = 30
    Site7 = 31

class Joint(object):
    ''' Represent a single joint. '''
    def __init__(self, x, y, z, joint_type):
        self.x = x
        self.y = y
        self.z = z
        self.joint_type = joint_type

class Body(object):
    ''' Represent a full body structure for Human 3.6m mocap data. '''

    def __init__(self, x_data_scale=5800, y_data_scale=5800, z_data_scale=2400):
        ''' Initialize body object. '''

        self._joints = np.zeros(shape=(self.joint_count, 3), dtype=np.float32)
        self._angles = []

        # The max range in x, y and z from Human3.6m dataset.
        self._x_data_scale = x_data_scale
        self._y_data_scale = y_data_scale
        self._z_data_scale = z_data_scale

        bone_starts = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        bone_ends = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1

        self._bones = []
        for i in range(len(bone_starts)):
            self._bones.append((bone_starts[i], bone_ends[i], self._location_name(bone_starts[i], bone_ends[i])))

    @property
    def joint_count(self):
        return 32

    @property
    def joints(self):
        return self._joints

    @property
    def bones(self):
        return self._bones

    def add_joints(self, joints):
        ''' Add 32 joints stored in an numpy array. '''
        self._joints = np.array(joints, dtype=np.float32)

    def as_numpy(self):
        return self._joints

    def normalize(self, x, y, z, at_depth):
        ''' Normalize from world space to [-1,-1] for each dimension. '''

        x_norm = x / self._x_data_scale
        y_norm = y / self._y_data_scale
        z_norm = z / self._z_data_scale

        return x_norm, y_norm, z_norm

    def unnormalize(self, x_norm, y_norm, z_norm, at_depth):
        ''' Return world space from normalized input. '''

        x = x_norm * self._x_data_scale
        y = y_norm * self._y_data_scale
        z = z_norm * self._z_data_scale

        return x, y, z
    
    def _location_name(self, start_joint, end_joint):
        name = 'center'
        start_joint_name = JointType(start_joint).name
        end_joint_name = JointType(end_joint).name

        if ('Right' in start_joint_name) or ('Right' in end_joint_name):
            name = 'right'
        elif  ('Left' in start_joint_name) or ('Left' in end_joint_name):
            name = 'left'

        return name