import tensorflow as tf
from . import vgg

def build(inputs, output_dims, model_name):
    '''
    A factory function to create models.

    Args:
        inputs: input variable.
        output_dims(int): number of classes.
        model_name(str): model name.

    Return:
        An instance of the model.
    '''
    return vgg.Vgg16(inputs, output_dims)

def read_model(model_path):
    '''
    Read a tensorflow graph def model.

    Args:
        model_path(str): the path to the model pb file.

    Return:
        A new GraphDef.
    '''
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        return graph_def
