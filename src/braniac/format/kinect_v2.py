import sys
import math
import numpy as np
from enum import Enum

class Sensor(object):
    ''' Sensor object contains all sensor specific properties '''

    def __init__(self):
        self._max_depth = 8. # 8 meters.
        self._horizontal_fov = 70.
        self._vertical_fov = 60.
        self._width = 512.
        self._height = 424.

        # Based on OpenNI (https://github.com/occipital/OpenNI2)
        self._xz_factor = math.tan(math.radians(self._horizontal_fov/2)) * 2.
        self._yz_factor = math.tan(math.radians(self._vertical_fov/2)) * 2.

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def horizontal_fov(self):
        return self._horizontal_fov

    @property
    def vertical_fov(self):
        return self._vertical_fov

    def get_horizontal_distance(self, depth):
        ''' Return the horizontal range at the provided depth '''
        return float(depth)*self._xz_factor

    def get_vertical_distance(self, depth):
        ''' Return the vertical range at the provided depth '''
        return float(depth)*self._yz_factor

    def project_3d_to_2d(self, pose3d, subject_id=0, camera_index=0):
        pose2d = np.empty(shape=(pose3d.shape[0], 2), dtype=np.float32)
        for i in range(pose3d.shape[0]):
            x, y = self._map_world_to_depth(pose3d[i, 0], pose3d[i, 1], pose3d[i, 2])
            pose2d[i, 0] = x
            pose2d[i, 1] = y
        return pose2d

    def _map_world_to_depth(self, x, y, z):
        ''' Map from word coordinate to 2D depth coordinate, based on OpenNI '''
        depth_x = (self._width/self._xz_factor) * (x/z) + self._width/2
        depth_y = self._height/2 - (self._height/self._yz_factor) * (y/z)
        return depth_x, depth_y

class JointType(Enum):
    ''' List of the joint types and their corresponding index '''
    Unknown = -1       # The type isn't available
    SpineBase = 0      # Base of the spine
    SpineMid = 1       # Middle of the spine
    Neck = 2           # Neck
    Head = 3           # Head
    ShoulderLeft = 4   # Left shoulder
    ElbowLeft = 5      # Left elbow
    WristLeft = 6      # Left wrist
    HandLeft = 7       # Left hand
    ShoulderRight = 8  # Right shoulder
    ElbowRight = 9     # Right elbow
    WristRight = 10    # Right wrist
    HandRight = 11     # Right hand
    HipLeft = 12       # Left hip
    KneeLeft = 13      # Left knee
    AnkleLeft = 14     # Left ankle
    FootLeft = 15      # Left foot
    HipRight = 16      # Right hip
    KneeRight = 17     # Right knee
    AnkleRight = 18    # Right ankle
    FootRight = 19     # Right foot
    SpineShoulder = 20 # Spine at the shoulder
    HandTipLeft = 21   # Tip of the left hand
    ThumbLeft = 22     # Left thumb
    HandTipRight = 23  # Tip of the right hand
    ThumbRight = 24    # Right thumb

class Joint(object):
    ''' Represent a single joint. '''
    def __init__(self, type_index=-1):
        self.joint_type = JointType(type_index)

        self.x = 0
        self.y = 0
        self.z = 0

        self.depth_x = 0
        self.depth_y = 0

        self.color_x = 0
        self.color_y = 0

        self.orientation_w = 0
        self.orientation_x = 0
        self.orientation_y = 0
        self.orientation_z = 0

        self.tracking_state = 0

class Body(object):
    ''' Represent a full body structure for Kinect V2. '''
    def __init__(self, body_id):
        ''' Initialize body object. '''
        self.id = body_id
        self.cliped_edges = 0
        self.hand_left_confidence = 0
        self.hand_left_state = 0
        self.hand_right_confidence = 0
        self.hand_right_state = 0
        self.restricted = 0
        self.lean_x = 0
        self.lean_y = 0
        self._joints = []
        self._np_joints = None
        self.tracking_state = 0
        self._sensor = Sensor()

        # keep track of some statistics.
        self.min_depth_x = sys.float_info.max
        self.min_depth_y = sys.float_info.max
        self.max_depth_x = sys.float_info.min
        self.max_depth_y = sys.float_info.min
        self.min_color_x = sys.float_info.max
        self.min_color_y = sys.float_info.max
        self.max_color_x = sys.float_info.min
        self.max_color_y = sys.float_info.min
        self.max_x = sys.float_info.min
        self.min_x = sys.float_info.max
        self.max_y = sys.float_info.min
        self.min_y = sys.float_info.max
        self.max_z = sys.float_info.min
        self.min_z = sys.float_info.max

        bone_ends = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
        self._bones = []
        for i in range(len(bone_ends)):
            self._bones.append((i, bone_ends[i], self._location_name(i, bone_ends[i])))

    @property
    def joint_count(self):
        return 25

    @property
    def joints(self):
        return self._joints

    @property
    def bones(self):
        return self._bones

    def add_joints(self, joints):
        self._joints = joints

        ''' Pack the joint (x, y, z) coordinate into numpy array '''
        np_joints = np.empty(shape=(self.joint_count, 3), dtype=np.float32)
        for i in range(self.joint_count):
            self._update_states(joints[i])
            np_joints[i, 0] = self.joints[i].x
            np_joints[i, 1] = self.joints[i].y
            np_joints[i, 2] = self.joints[i].z
        self._np_joints = np_joints

    def as_numpy(self):
        return self._np_joints

    def normalize(self, x, y, z, at_depth):
        ''' Normalize from world space to [-1,-1] for each dimension. '''

        z_scale = at_depth
        x_scale = self._sensor.get_horizontal_distance(at_depth)
        y_scale = self._sensor.get_vertical_distance(at_depth)

        x_norm = x / x_scale
        y_norm = y / y_scale
        z_norm = (z - z_scale / 2.) / z_scale

        return x_norm, y_norm, z_norm

    def unnormalize(self, x_norm, y_norm, z_norm, at_depth):
        ''' Return world space from normalized input. '''

        z_scale = at_depth
        x_scale = self._sensor.get_horizontal_distance(at_depth)
        y_scale = self._sensor.get_vertical_distance(at_depth)

        x = x_norm * x_scale
        y = y_norm * y_scale
        z = (z_norm * z_scale) + z_scale / 2.

        return x, y, z
    
    def _update_states(self, joint):
        ''' Add new joint to Body and keep track of some statistics. '''
        self.min_x = min(joint.x, self.min_x)
        self.min_y = min(joint.y, self.min_y)
        self.min_z = min(joint.z, self.min_z)
        self.max_x = max(joint.x, self.max_x)
        self.max_y = max(joint.y, self.max_y)
        self.max_z = max(joint.z, self.max_z)

        self.min_depth_x = min(joint.depth_x, self.min_depth_x)
        self.min_depth_y = min(joint.depth_y, self.min_depth_y)
        self.max_depth_x = max(joint.depth_x, self.max_depth_x)
        self.max_depth_y = max(joint.depth_y, self.max_depth_y)

        self.min_color_x = min(joint.color_x, self.min_color_x)
        self.min_color_y = min(joint.color_y, self.min_color_y)
        self.max_color_x = max(joint.color_x, self.max_color_x)
        self.max_color_y = max(joint.color_y, self.max_color_y)

    def _location_name(self, start_joint, end_joint):
        name = 'center'
        start_joint_name = JointType(start_joint).name
        end_joint_name = JointType(end_joint).name

        if ('Right' in start_joint_name) or ('Right' in end_joint_name):
            name = 'right'
        elif  ('Left' in start_joint_name) or ('Left' in end_joint_name):
            name = 'left'

        return name