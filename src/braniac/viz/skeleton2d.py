import sys
import math
import numpy as np
from PIL import Image, ImageDraw

class Skeleton2D(object):
    '''
    Project 3D skeleton into 2D plan and render the result into an image object.
    '''
    def __init__(self, sensor, body):
        '''
        Initialize skeleton 2D object and store the source of the data.

        Args:
            sensor(Sensor): Sensor contain the intrinsic properties of the source of the data.
            body(Body): Body object contain mapping information for body joints depend on the device source.
        '''        
        self._sensor = sensor
        self._body = body
        self._width = float(self._sensor.width)
        self._height = float(self._sensor.height)

    def draw_to_file(self, multiple_sequence_of_skeleton, subject_id, image_path):
        '''
        Render multiple sequences of skeletons into 2D image.

        Args:
            multiple_sequence_of_skeleton(List of numpy.array): List of skeleton sequences.
            image_path(str): path to the output image file.
        '''
        self.draw_to_image(multiple_sequence_of_skeleton, subject_id).save(image_path)

    def draw_to_video_file(self, sequence_of_skeletons, video_path):
        '''
        Render a sequence of skeletons into 2D images.

        Args:
            sequence_of_skeletons(List of numpy.array): skeleton sequence.
            video_path(str): path to the output video file.
        '''
        from moviepy.editor import ImageSequenceClip
        images = self.draw_to_images(sequence_of_skeletons)
        video = ImageSequenceClip(images, fps=15)
        video.write_videofile(video_path)

    def draw_to_image(self, multiple_sequence_of_skeleton, subject_id):
        '''
        Render multiple sequences of skeletons into 2D image.

        Args:
            multiple_sequence_of_skeleton(List of numpy.array): List of skeleton sequences.
        '''                

        num_of_sequences = len(multiple_sequence_of_skeleton)
        camera_index = 1
        image_width = 0
        image_height = 0
        for sequence_of_skeleton in multiple_sequence_of_skeleton:
            num_of_skeletons = sequence_of_skeleton.shape[0]
            num_of_joints = sequence_of_skeleton.shape[1]

            margin = 12
            pose2d_max = np.array([0, 0])
            pose2d_min = np.array([sys.maxsize, sys.maxsize])

            for skeleton_index in range(num_of_skeletons):
                pose2d = self._sensor.project_3d_to_2d(sequence_of_skeleton[skeleton_index, :, :], subject_id, camera_index)
                pose2d_max = np.maximum(np.amax(pose2d, axis=0), pose2d_max)
                pose2d_min = np.minimum(np.amin(pose2d, axis=0), pose2d_min)

            image_width = max(image_width, int(math.ceil((pose2d_max[0] - pose2d_min[0]) + margin)))
            image_height = max(image_height, int(math.ceil((pose2d_max[1] - pose2d_min[1]) + margin)))

        image = Image.new('RGB', (image_width*num_of_skeletons, image_height*num_of_sequences), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        y_offset = 0
        for sequence_of_skeleton in multiple_sequence_of_skeleton:
            num_of_skeletons = sequence_of_skeleton.shape[0]
            num_of_joints = sequence_of_skeleton.shape[1]

            for skeleton_index in range(num_of_skeletons):
                joints = []
                pose2d = self._sensor.project_3d_to_2d(sequence_of_skeleton[skeleton_index, :, :], subject_id, camera_index)
                for i in range(num_of_joints):
                    x = min(max(pose2d[i, 0], 0.), self._width)
                    y = min(max(pose2d[i, 1], 0.), self._height)

                    x = x - pose2d_min[0] + margin/2
                    y = y - pose2d_min[1] + margin/2
                    x += skeleton_index * image_width

                    x = int(round(x))
                    y = int(round(y))
                    joints.append((x, y + y_offset))

                    x0 = max(x-2, skeleton_index * image_width)
                    x1 = min(x+2, (skeleton_index+1) * image_width)
                    y0 = max(y-2, 0)
                    y1 = min(y+2, image_height)

                    draw.ellipse([x0, y0 + y_offset, x1, y1 + y_offset], fill=(0, 0, 0), outline=(0, 0, 0))

                for bone in self._body.bones:
                    start = joints[bone[0]]
                    end = joints[bone[1]]
                    location = bone[2]

                    if location == 'center':
                        draw.line([start, end], fill=(128,0,0), width=2)
                    elif location == 'right':
                        draw.line([start, end], fill=(0,128,0), width=3)
                    else:
                        draw.line([start, end], fill=(0,0,128), width=3)
                        
            y_offset += image_height

        return image

    def draw_to_images_1(self, sequence_of_skeletons):
        '''
        Render a sequence of skeletons to multiple 2D images, one per skeleton.

        Args:
            sequence_of_skeleton(List of numpy.array): A skeleton sequence.
        '''                

        images = []
        num_of_skeletons = sequence_of_skeletons.shape[0]
        num_of_joints = sequence_of_skeletons.shape[1]

        for skeleton_index in range(num_of_skeletons):
            image = Image.new('RGB', (int(self._width), int(self._height)), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            joints = []
            for i in range(num_of_joints):
                x = sequence_of_skeletons[skeleton_index, i, 0]
                y = sequence_of_skeletons[skeleton_index, i, 1]
                z = sequence_of_skeletons[skeleton_index, i, 2]

                x, y, z = self._body.unnormalize(x, y, z, 5.)
                depth_x, depth_y = self._sensor.map_world_to_depth(x, y, z)
                depth_x = min(max(depth_x, 0.), self._width)
                depth_y = min(max(depth_y, 0.), self._height)

                depth_x = int(round(depth_x))
                depth_y = int(round(depth_y))
                joints.append((depth_x, depth_y))

                x0 = max(depth_x-2, 0)
                x1 = min(depth_x+2, int(self._width))
                y0 = max(depth_y-2, 0)
                y1 = min(depth_y+2, int(self._height))

                draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0), outline=(0, 0, 0))

            for bone in self._body.bones:
                start = joints[bone[0]]
                end = joints[bone[1]]
                draw.line([start, end], fill=(128,0,0), width=2)
                
            images.append(np.asarray(image))

        return images