import tensorflow as tf

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
