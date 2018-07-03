import numpy as np
import tensorflow as tf

def one_hot(indices, depth, axis=-1, dtype=np.float32):
    ''' Compute one hot from indices at a specific axis '''

    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis = rank + axis + 1

    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,)*len(ls)+depth_range.shape+(1,)*len(rs))
    values = np.reshape(values, ls+(1,)+rs)
    return np.asarray(targets == values, dtype=dtype)

def bone_loss(prev_sequence, next_sequence, body):
    ''' Compare the bone size of the predicted sequence from the last skeleton
        in the input sequence. '''
    num_of_skeletons = prev_sequence.shape[1]
    num_of_joints = prev_sequence.shape[2]

    loss = 0.
    for skeleton_index in range(num_of_skeletons):
        for bone in body.bones:
            i = bone[0]
            j = bone[1]

            x1 = prev_sequence[:, 0, i, 0]
            y1 = prev_sequence[:, 0, i, 1]
            z1 = prev_sequence[:, 0, i, 2]

            x2 = prev_sequence[:, 0, j, 0]
            y2 = prev_sequence[:, 0, j, 1]
            z2 = prev_sequence[:, 0, j, 2]

            bone_length_ref = tf.sqrt(tf.squared_difference(x2, x1) + tf.squared_difference(y2, y1) + tf.squared_difference(z2, z1))

            x1 = next_sequence[:, skeleton_index, i, 0]
            y1 = next_sequence[:, skeleton_index, i, 1]
            z1 = next_sequence[:, skeleton_index, i, 2]

            x2 = next_sequence[:, skeleton_index, j, 0]
            y2 = next_sequence[:, skeleton_index, j, 1]
            z2 = next_sequence[:, skeleton_index, j, 2]

            bone_length = tf.sqrt(tf.squared_difference(x2, x1) + tf.squared_difference(y2, y1) + tf.squared_difference(z2, z1))

            loss += tf.reduce_sum(tf.abs(bone_length_ref - bone_length) / (bone_length_ref + 0.00001))
    return loss

def attention(inputs, 
              activation=None, 
              attention_len=None, 
              kernel_initializer=None,
              bias_initializer=None):
              
    num_neurons = inputs.shape[2].value
    if attention_len == None:
        attention_len = num_neurons

    # Trainable parameters
    Watt = tf.get_variable("Watt", shape=[num_neurons, attention_len], initializer=kernel_initializer)
    batt = tf.get_variable("batt", shape=[attention_len], initializer=bias_initializer)
    uatt = tf.get_variable("uatt", shape=[attention_len], initializer=kernel_initializer)

    # (Batch,Seq,Neuron) * (Neuron,Att) --> (Batch,Seq,Att)
    v = tf.einsum('bsn,na->bsa', inputs, Watt) + batt
    if activation != None:
        v = activation(v)

    # (Batch,Seq,Att) * (Att) --> (Batch, Seq)
    vu = tf.einsum('bsa,a->bs', v, uatt)
    alpha = tf.nn.softmax(vu, name='alpha')

    return tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)

def create_rnn_cell(cell_type, num_neurons, use_residual=False):
    ''' Create RNN cell depend on the provided type '''
    cell = None
    if cell_type == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell(num_neurons)
    elif cell_type == "lstmp":
        cell = tf.nn.rnn_cell.LSTMCell(num_neurons, use_peepholes=True)        
    elif cell_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(num_neurons)
    elif cell_type == "norm_lstm":
        cell = tf.nn.rnn_cell.LayerNormBasicLSTMCell(num_neurons)
    else:
        raise Exception("Unsupported cell type.")

    if use_residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    return cell

def create_rnn_model(num_layers, cell_type, num_neurons, use_residual=False):
    ''' Create RNN model '''
    if num_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell([create_rnn_cell(cell_type, num_neurons, use_residual) for _ in range(num_layers)])
    else:
        return create_rnn_cell(cell_type, num_neurons, use_residual)

def generate_random(rand_type, params, shape):
    '''
    Generate a random number with the specified shape.
    '''
    if rand_type == 'normal':
        return np.random.normal(params['mean'], params['std'], size=shape)
    elif rand_type == 'uniform':
        return np.random.uniform(params['low'], params['high'], size=shape)
    else:
        raise ValueError('Unknown random type.')

def generate_skeletons_with_prob(sess, g, d_real_prob, d_fake_prob, data_preprocessing, 
                                 d_inputs, g_inputs, g_z):
    '''
    Generate multiple future skeleton poses for each `z` in `g_z`.
    '''
    def generate(input_data, input_past_data, z_data_p, batch_index=0):
        skeleton_data = []
        d_is_sequence_probs = []

        prob = sess.run(d_real_prob.prob, feed_dict={d_inputs:input_data})
        
        skeleton_data.append(data_preprocessing.unnormalize(input_data[batch_index, :, :, :]))
        d_is_sequence_probs.append(prob[batch_index][0])
        for z_value_p in z_data_p:
            pred, prob = sess.run([g.output, d_fake_prob.prob], feed_dict={g_inputs:input_past_data, g_z:z_value_p})
            inout_pred = np.concatenate((input_past_data, pred), axis=1)
            skeleton_data.append(data_preprocessing.unnormalize(inout_pred[batch_index, :, :, :]))
            d_is_sequence_probs.append(prob[batch_index][0])
        
        return skeleton_data, d_is_sequence_probs
    
    return generate