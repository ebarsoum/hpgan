import numpy as np
import tensorflow as tf

from .. import nn as nn

class RNNDiscriminator(object):
    '''
    GAN discriminator network that discriminate between real future frames versus
    synthetic predicted frames by the generator. The discriminator uses recurrent
    network.
    '''
    def __init__(self, inputs, inputs_depth, sequence_length, 
                 use_attention=False, use_residual=False, cell_type='gru', output_category_dims=None,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                 reuse=False, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
            inputs_depth(int): input embed size.
            sequence_length(int): the length of the input sequence.
            reuse(bool): True to reuse model parameters from a previously created model.
            scope(str): Prepend variable scope with `scope`.
        '''
        self._reuse = reuse
        self._use_attention = use_attention
        self._use_residual=use_residual
        self._bias_initializer=bias_initializer
        self._kernel_initializer=kernel_initializer
        self._cell_type = cell_type
        self._num_neurons = 1024
        self._num_layers = 2
        self._sequence_length = sequence_length
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output_dims = 1
        self._output_category_dims = output_category_dims
        self._output = None
        self._output_category = None        
        self._prob = None
        self._parameters = []
        self._weights = []
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"
        self._build(inputs)

    @property
    def output(self):
        ''' Raw output of the network '''
        return self._output

    @property
    def output_category(self):
        ''' Raw classification output of the network '''
        return self._output_category

    @property
    def prob(self):
        ''' Probalistic output of the network '''
        return self._prob

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs):
        '''
        Construct a discriminator model.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
        '''
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            cell = nn.create_rnn_model(self._num_layers, self._cell_type, self._num_neurons)
            outputs, state = tf.nn.dynamic_rnn(cell, d_inputs, dtype=tf.float32)

            if self._use_attention:
                last = nn.attention(outputs,
                                    kernel_initializer=self._kernel_initializer,
                                    bias_initializer=self._bias_initializer)
            else:
                outputs = tf.transpose(outputs, [1, 0, 2])
                last = tf.gather(outputs, int(outputs.shape[0]) - 1)

            base = tf.layers.dense(inputs=last,
                                   units=self._num_neurons,
                                   activation=tf.nn.relu,
                                   name="fc1")

            self._output = tf.layers.dense(inputs=base,
                                           units=self._output_dims,
                                           activation=None,
                                           name="output")

            if self._output_category_dims != None:
                self._output_category = tf.layers.dense(inputs=base,
                                                        units=self._output_category_dims,
                                                        activation=None,
                                                        name="output_categories")

            # Wo = tf.get_variable("Wo", shape=[self._num_neurons, self._output_dims], initializer=self._kernel_initializer)
            # bo = tf.get_variable("bo", shape=[self._output_dims], initializer=self._bias_initializer)

            # self._output = tf.matmul(last, Wo) + bo
            self._prob = tf.nn.sigmoid(self._output)

            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('Wo:0'))]

class NNResidualDiscriminator(object):
    '''
    GAN discriminator network that discriminate between real future frames versus
    synthetic predicted frames by the generator. The discriminator uses feedforward
    neural network with residual connection.
    '''
    def __init__(self, 
                 inputs, 
                 inputs_depth,
                 sequence_length, 
                 activation=tf.nn.relu,
                 num_residual_blocks=3,
                 bias_initializer=tf.constant_initializer(0.),
                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                 reuse=False, 
                 scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
            inputs_depth(int): input embed size.
            sequence_length(int): the length of the input sequence.
            activation: activiation function to use.
            bias_initializer: initializer for the bias value.
            kernel_initializer: initializer for the `W` parameters.
            reuse(bool): True to reuse model parameters from a previously created model.
            scope(str): Prepend variable scope with `scope`.
        '''
        self._activation = activation
        self._bias_initializer=bias_initializer
        self._kernel_initializer=kernel_initializer
        self._reuse = reuse
        self._num_neurons = 512
        self._num_residual_blocks = num_residual_blocks
        self._sequence_length = sequence_length
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output_dims = 1
        self._output = None
        self._prob = None
        self._parameters = []
        self._weights = []
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"

        self._build(inputs)

    @property
    def output(self):
        ''' Raw output of the network '''
        return self._output

    @property
    def prob(self):
        ''' Probalistic output of the network '''
        return self._prob

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build_residual_unit(self, inputs, residual_index):
        '''
        Build a single residual unit.

        Args:
            inputs: input to the resnet unit.
        '''
        net1 = tf.layers.dense(inputs=inputs,
                               units=self._num_neurons,
                               activation=self._activation, 
                               name="resnet1_{}".format(residual_index+1), 
                               reuse=self._reuse)
        net2 = tf.layers.dense(inputs=net1,
                               units=inputs.shape[-1],
                               activation=None, 
                               name="resnet2_{}".format(residual_index+1), 
                               reuse=self._reuse)
        return  self._activation(net2 + inputs)

    def _build(self, inputs):
        '''
        Construct a discriminator model.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
        '''
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])
            net = tf.layers.dense(inputs=net, 
                                  units=self._num_neurons, 
                                  activation=self._activation, 
                                  name="fc1", 
                                  reuse=self._reuse)

            residual_index = 0
            for _ in range(self._num_residual_blocks):
                net = self._build_residual_unit(net, residual_index)
                residual_index += 1

            self._output = tf.layers.dense(inputs=net, 
                                           units=self._output_dims, 
                                           name="fc2", 
                                           reuse=self._reuse)
            self._prob = tf.nn.sigmoid(self._output)

            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('kernel:0'))]

class NNDiscriminator(object):
    '''
    GAN discriminator network that discriminate between real future frames versus
    synthetic predicted frames by the generator. The discriminator uses feedforward
    neural network.
    '''
    def __init__(self, inputs, inputs_depth, sequence_length, reuse=False, scope=""):
        '''
        Initialize the discriminator network.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
            inputs_depth(int): input embed size.
            sequence_length(int): the length of the input sequence.
            reuse(bool): True to reuse model parameters from a previously created model.
            scope(str): Prepend variable scope with `scope`.
        '''
        self._reuse = reuse
        self._num_neurons = 512
        self._num_layers = 3
        self._sequence_length = sequence_length
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output_dims = 1
        self._output = None
        self._prob = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        if scope.strip() == "":
            self._scope = scope
        else:
            self._scope = scope + "/"

        self._build(inputs)

    @property
    def output(self):
        ''' Raw output of the network '''
        return self._output

    @property
    def prob(self):
        ''' Probalistic output of the network '''
        return self._prob

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs):
        '''
        Construct a discriminator model.

        Args:
            inputs(tf.placeholder): The input variable containing the data.
        '''
        with tf.variable_scope(self._scope + self.__class__.__name__, reuse=self._reuse) as vs:
            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])

            layer_index = 0
            for _ in range(self._num_layers):
                net = tf.layers.dense(inputs=net, units=self._num_neurons, activation=tf.nn.relu, name="fc{}".format(layer_index+1), reuse=self._reuse)
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            self._output = tf.layers.dense(inputs=net, units=self._output_dims, name="fc{}".format(layer_index+1), reuse=self._reuse)
            self._prob = tf.nn.sigmoid(self._output)

            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('kernel:0'))]

class RNNGenerator(object):
    '''
    GAN generator network that predict a single future skeleton pose using RNN network.
    '''
    def __init__(self, inputs, inputs_depth, reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = inputs.shape[0] # batch_size
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        self._build(inputs)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
        '''
        with tf.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            num_neurons = 256
            num_layers = 2

            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape, stddev=self._stddev))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            cell = tf.contrib.rnn.LSTMCell(num_neurons)
            # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=0.5)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

            # Getting the last output
            outputs, state = tf.nn.dynamic_rnn(cell, d_inputs, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            last = tf.gather(outputs, int(outputs.shape[0]) - 1)

            Wo = tf.get_variable("Wo", initializer=tf.truncated_normal([num_neurons, self._element_shape[0]], stddev=self._stddev))
            bo = tf.get_variable("bo", initializer=tf.constant(0., shape=[self._element_shape[0]]))

            pred = tf.matmul(last, Wo) + bo
            pred = tf.reshape(pred, pred.shape[:1].as_list() + [1] + self._inputs_shape[2:].as_list())

            self._output = tf.tanh(pred)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('Wo:0'))]

class NNGenerator(object):
    '''
    GAN generator network that predict a single future skeleton pose using feedforward neural network.
    '''
    def __init__(self, inputs, inputs_depth, z, reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = inputs.shape[0] # batch_size
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = [np.prod(inputs.shape[2:].as_list()), inputs_depth]
        self._output = None
        self._parameters = []
        self._weights = []
        self._stddev = 0.001
        self._build(inputs, z)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs, z):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
        '''
        with tf.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            num_neurons = 1024
            num_layers = 3

            Wi = tf.get_variable("Wi", initializer=tf.truncated_normal(self._element_shape, stddev=self._stddev))
            inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])

            d_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            d_inputs.set_shape([None, inputs.shape[1].value, self._element_shape[-1]]) # https://github.com/tensorflow/tensorflow/issues/6682

            net = tf.reshape(d_inputs, [-1, np.prod(d_inputs.shape[1:].as_list())])

            for _ in range(num_layers):
                net = tf.layers.dense(inputs=net, units=num_neurons, activation=tf.nn.relu, reuse=self._reuse)
                net = tf.layers.dropout(inputs=net, rate=0.5)

            pred = tf.layers.dense(inputs=net, units=self._element_shape[0], reuse=self._reuse)
            pred = tf.reshape(pred, pred.shape[:1].as_list() + [1] + self._inputs_shape[2:].as_list())

            self._output = tf.tanh(pred)
            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or v.name.endswith('weights:0') or v.name.endswith('kernel:0'))]

class SequenceToSequenceGenerator(object):
    '''
    GAN generator network that predict future skeleton poses using sequernce to sequence network.
    '''
    def __init__(self, inputs, inputs_depth, z, input_sequence_length, output_sequence_length, 
                 cell_type='gru', project_to_rnn_output=False, reverse_input=False,
                 use_attention=False, use_residual=False,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                 reuse=False):
        '''
        Initialize the generative network.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            inputs_depth(int): input embed size.
            z(tf.placeholder, optional): A random generated input vector used as input.
            input_sequence_length(int): the length of the input sequence.
            output_sequence_length(int): the length of the resulted sequence.
            cell_type(str): The type of cell to use for the encode and decoder.
            project_to_rnn_output(bool): project the input to the number of hidden unit in the RNN.
            reverse_input(bool): reverse the input sequence before feeding it to the network.
            use_attention(bool): true to use attention instead of the last state of the encoder.
            use_residual(bool): use resent like structure for the recurrent.
            bias_initializer: initializer for the bias value.
            kernel_initializer: initializer for the `W` parameters.            
            reuse(bool): True to reuse model parameters from a previously created model.
        '''
        self._reuse = reuse
        self._batch_size = tf.shape(inputs)[0] # batch_size
        self._input_sequence_length = input_sequence_length
        self._output_sequence_length = output_sequence_length
        self._inputs_depth = inputs_depth
        self._inputs_shape = inputs.shape
        self._element_shape = inputs.shape[2:].as_list()
        self._output = None
        self._parameters = []
        self._weights = []
        self._num_neurons = 1024
        self._num_layers = 2
        self._num_nn_layers = 2
        self._cell_type = cell_type
        self._bias_initializer = bias_initializer
        self._kernel_initializer = kernel_initializer
        self._reccurent_bias_initializer = None
        self._reccurent_kernel_initializer = None
        self._project_to_rnn_output = project_to_rnn_output
        self._use_attention = use_attention
        self._use_residual = use_residual

        if self._use_residual:
            self._project_to_rnn_output = True

        # Similar to tf.zeros but support variable batch size.
        if self._project_to_rnn_output:
            self._zeros_input = tf.fill(tf.stack([tf.shape(inputs)[0], self._num_neurons]), 0.0)
        else:
            self._zeros_input = tf.fill(tf.stack([tf.shape(inputs)[0], self._inputs_depth]), 0.0)

        if reverse_input:
            inputs = tf.reverse(inputs, axis=[1])
        self._build(inputs, z)

    @property
    def output(self):
        return self._output

    @property
    def parameters(self):
        ''' All trainable parameters '''
        return self._parameters

    @property
    def weights(self):
        ''' Weights only parameters for regularization '''
        return self._weights

    def _build(self, inputs, z):
        '''
        Construct a generative model.

        Args:
            inputs(tf.placeholder): The input variable containing current data.
            z(tf.placeholder): A vector containss the randomly generated latent data.
        '''
        with tf.variable_scope(self.__class__.__name__, reuse=self._reuse) as vs:

            outputs, encoder_state = self._build_encoder(inputs, z)

            first_input = outputs[:, -1, :] # [batch, sequence, elements]
            if self._use_attention:
                encoder_state = nn.attention(outputs, 
                                             kernel_initializer=self._kernel_initializer,
                                             bias_initializer=self._bias_initializer)
            self._output = self._build_decoder(first_input, z, encoder_state)

            self._parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name+'/')
            self._weights = [v for v in self._parameters if (v.name.endswith('Wi:0') or \
                                                             v.name.endswith('weights:0') or \
                                                             v.name.endswith('Wo:0') or \
                                                             v.name.endswith('Wsi:0') or \
                                                             ('Wzi' in v.name) or \
                                                             ('Wzci' in v.name) or \
                                                             ('Wzhi' in v.name))]

    def _create_rnn_model(self):
        ''' Create RNN model '''
        return nn.create_rnn_model(self._num_layers, 
                                   self._cell_type, 
                                   self._num_neurons, 
                                   use_residual=self._use_residual)

    def _input_projection(self, inputs):
        ''' Project each skeleton pose to the encoder. '''

        inputs = tf.reshape(inputs, shape=[-1, inputs.shape[1].value]+[np.prod(inputs.shape[2:].as_list())])
        if self._project_to_rnn_output:
            net = inputs
            layer_index = 0
            num_neurons = self._num_neurons // (self._num_nn_layers+1)
            for i in range(self._num_nn_layers):
                net = tf.layers.dense(inputs=net, 
                                      units=(i+1)*num_neurons,
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu,
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            encoder_inputs = tf.layers.dense(inputs=net, 
                                             units=self._num_neurons,
                                             kernel_initializer=self._kernel_initializer,
                                             bias_initializer=self._bias_initializer,
                                             activation=None, 
                                             name="fc{}".format(layer_index+1))
            encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._num_neurons])
        else:
            Wi = tf.get_variable("Wi", shape=[np.prod(self._element_shape), self._inputs_depth], initializer=self._kernel_initializer)
            bi = tf.get_variable("bi", shape=[self._inputs_depth], initializer=self._bias_initializer)

            encoder_inputs = tf.tensordot(inputs, Wi, axes=[[2], [0]])
            encoder_inputs.set_shape([inputs.shape[0].value, inputs.shape[1].value, self._inputs_depth]) # https://github.com/tensorflow/tensorflow/issues/6682
            encoder_inputs = encoder_inputs + bi

        return encoder_inputs

    def _output_projection(self, outputs):
        ''' Project each decoder output back to skeleton pose. '''

        if self._project_to_rnn_output:
            net = outputs
            layer_index = 0
            for i in range(self._num_nn_layers):
                net = tf.layers.dense(inputs=net, 
                                      units=int(self._num_neurons/(i+1)),
                                      kernel_initializer=self._kernel_initializer,
                                      bias_initializer=self._bias_initializer,
                                      activation=tf.nn.relu, 
                                      name="fc{}".format(layer_index+1))
                # net = tf.layers.dropout(inputs=net, rate=0.5)
                layer_index += 1

            pred = tf.layers.dense(inputs=net, 
                                   units=np.prod(self._element_shape),
                                   kernel_initializer=self._kernel_initializer,
                                   bias_initializer=self._bias_initializer,
                                   activation=None, 
                                   name="fc{}".format(layer_index+1))

            pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)])
            pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())
        else:
            Wo = tf.get_variable("Wo", shape=[self._num_neurons, np.prod(self._element_shape)], initializer=self._kernel_initializer)
            bo = tf.get_variable("bo", shape=[np.prod(self._element_shape)], initializer=self._bias_initializer)

            pred = tf.tensordot(outputs, Wo, axes=[[2], [0]])
            pred.set_shape([outputs.shape[0].value, self._output_sequence_length, np.prod(self._element_shape)]) # https://github.com/tensorflow/tensorflow/issues/6682
            pred = pred + bo
            pred = tf.reshape(pred, shape=[-1, pred.shape[1].value] + self._inputs_shape[2:].as_list())

        return pred

    def _build_encoder(self, inputs, z):
        ''' Build the encoder part of the generative mode. '''
        with tf.variable_scope("encoder", reuse=self._reuse):
            encoder_inputs = self._input_projection(inputs)
            cell = self._create_rnn_model()
            outputs, state = tf.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32)

            return outputs, state

    def _build_decoder(self, first_input, z, encoder_state):
        '''
        Build the decoder part of the generative mode. It can decode based on the initial state without
        the need of future_inputs.

        Args:
            first_input(tf.placeholder, optional): each cell takes input form the output of the previous cell,
                                                   except first cell. first_input is used for the first cell.
            z(tf.placeholder, optional): random vector in order to sample multiple predictions from the 
                                         same input.
            encoder_state(cell state): the last state of the encoder.

        Return:
            The output of the network.
        '''
        with tf.variable_scope("decoder", reuse=self._reuse):
            cell = self._create_rnn_model()
            outputs, _ = self._dynamic_rnn_decoder(cell, first_input, z, encoder_state, self._output_sequence_length)
            return self._output_projection(outputs)

    def _dynamic_rnn_decoder(self, cell, first_input, z, encoder_state, sequence_length, time_major=False, dtype=tf.float32):
        ''' Unroll the RNN decoder '''
        if not self._project_to_rnn_output:
            # From output state to input embed.
            Wsi = tf.get_variable("Wsi", 
                                  shape=[self._num_neurons, self._inputs_depth], 
                                  initializer=self._kernel_initializer)

        if first_input is None:
            first_input = self._zeros_input

        first_input = first_input if self._project_to_rnn_output else tf.matmul(first_input, Wsi)

        if z is not None:
            is_tuple = isinstance(encoder_state[0], tf.contrib.rnn.LSTMStateTuple) if (self._num_layers > 1) else isinstance(encoder_state, tf.contrib.rnn.LSTMStateTuple)
            if is_tuple:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzhi = tf.get_variable("Wzhi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                        Wzci = tf.get_variable("Wzci{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].c.shape.as_list()[-1]], initializer=self._kernel_initializer)                        
                        states.append(tf.contrib.rnn.LSTMStateTuple(encoder_state[i].c + tf.matmul(z, Wzci), encoder_state[i].h + tf.matmul(z, Wzhi)))
                    encoder_state = tuple(states)
                else:
                    Wzhi = tf.get_variable("Wzhi", shape=[z.shape.as_list()[-1], encoder_state.h.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    Wzci = tf.get_variable("Wzci", shape=[z.shape.as_list()[-1], encoder_state.c.shape.as_list()[-1]], initializer=self._kernel_initializer)      
                    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state.c + tf.matmul(z, Wzci), encoder_state.h + tf.matmul(z, Wzhi))
            else:
                if self._num_layers > 1:
                    states = []
                    for i in range(self._num_layers):
                        Wzi = tf.get_variable("Wzi{}".format(i), shape=[z.shape.as_list()[-1], encoder_state[i].shape.as_list()[-1]], initializer=self._kernel_initializer)
                        states.append(encoder_state[i] + tf.matmul(z, Wzi))
                    encoder_state = tuple(states)
                else:
                    Wzi = tf.get_variable("Wzi", shape=[z.shape.as_list()[-1], encoder_state.shape.as_list()[-1]], initializer=self._kernel_initializer)
                    encoder_state = encoder_state + tf.matmul(z, Wzi)

        def loop_fn_init(time):
            elements_finished = (sequence_length <= 0)
            next_input = first_input
            next_cell_state = encoder_state
            emit_output = None
            next_loop_state = None

            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn_next(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            next_cell_state = cell_state

            elements_finished = (time >= sequence_length)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: self._zeros_input,
                lambda: cell_output if self._project_to_rnn_output else tf.matmul(cell_output, Wsi))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                return loop_fn_init(time)
            else:
                return loop_fn_next(time, cell_output, cell_state, loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.stack()

        if not time_major:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, final_state

