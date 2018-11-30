# Adversarial Anomaly Detection
#
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================

import tensorflow as tf
import numpy as np
import util
import argparse

input_feature_dim = 150

width = 20
height = input_feature_dim

g_encoder_z_local_dim = 64
g_encoder_z_dim = g_encoder_z_local_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 128
g_encoder_layer2_dim = 64

lstm_hidden_size_layer1 = 128
lstm_hidden_size_layer2 = 64
lstm_sequence_length = 20

g_decoder_output_dim = input_feature_dim


def dense(x, n1, n2, scope='dense', initial_value=None):
    """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param scope: name of the entire dense layer.i.e, variable scope name.
        :param initial_value: initial value. list [weight, bias]
        :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(scope):
        if initial_value is None:
            weights = tf.get_variable("weights", shape=[n1, n2], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        else:
            weights = tf.get_variable("weights", initializer=initial_value[0])
            bias = tf.get_variable("bias", initializer=initial_value[1])

        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, bias=True):
    input_dims = input.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    # Define a variable scope for the conv layer
    with tf.variable_scope(scope):
        # Create filter weight variable
        conv_weight = tf.Variable(
            tf.truncated_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))
        # Create bias variable
        conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))
        # Define the convolution flow graph
        map = tf.nn.conv2d(input, conv_weight, strides=[1, stride_h, stride_w, 1], padding=padding)
        # Add bias to conv output
        if bias is True:
            map = tf.nn.bias_add(map, conv_bias)

        # Apply non-linearity (if asked) and return output
        activation = non_linear_fn(map)

        # print(activation.get_shape().as_list())
        return activation


def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    # Returns the height and width of the output of a deconvolution layer.
    batch_size, input_h, input_w, num_channels_in = input_dims
    filter_h, filter_w, num_channels_out  = filter_dims
    stride_h, stride_w = stride_dims

    # Compute the height in the output, based on the padding.
    if padding == 'SAME':
      out_h = input_h * stride_h
    elif padding == 'VALID':
      out_h = (input_h - 1) * stride_h + filter_h

    # Compute the width in the output, based on the padding.
    if padding == 'SAME':
      out_w = input_w * stride_w
    elif padding == 'VALID':
      out_w = (input_w - 1) * stride_w + filter_w

    return [batch_size, out_h, out_w, num_channels_out]


def deconv(input, batch_size, scope, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    # print(scope, 'in', input_dims)
    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    input_dims = [batch_size, input_dims[1], input_dims[2], input_dims[3]]
    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims
    # Let's step into this function
    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    # Define a variable scope for the deconv layer
    with tf.variable_scope(scope):
        # Create filter weight variable
        # Note that num_channels_out and in positions are flipped for deconv.
        deconv_weight = tf.Variable(
            tf.random_normal([filter_h, filter_w, num_channels_out, num_channels_in], stddev=0.1, dtype=tf.float32))
        # Create bias variable
        deconv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))

        # Define the deconv flow graph
        map = tf.nn.conv2d_transpose(input, deconv_weight, output_dims, strides=[1, stride_h, stride_w, 1], padding=padding)

        # Add bias to deconv output
        map = tf.nn.bias_add(map, deconv_bias)

        # Apply non-linearity (if asked) and return output
        activation = non_linear_fn(map)

        # print(scope, 'out', activation.get_shape().as_list())
        return activation


def max_pool(input, scope, filter_dims, stride_dims, padding='SAME'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        # Define the max pool flow graph and return output
        pool = tf.nn.max_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)

        return pool


def fc(input, scope, out_dim, non_linear_fn=None):
    assert (type(out_dim) == int)

    # Define a variable scope for the FC layer
    with tf.variable_scope(scope):
        input_dims = input.get_shape().as_list()
        # print(scope, 'in', input_dims)

        # the input to the fc layer should be flattened
        if len(input_dims) == 4:
            # for eg. the output of a conv layer
            batch_size, input_h, input_w, num_channels = input_dims
            # ignore the batch dimension
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input

        # Create weight variable
        fc_weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32))

        # Create bias variable
        fc_bias = tf.Variable(tf.zeros([out_dim], dtype=tf.float32))

        # Define FC flow graph
        output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)

        # print(scope, 'out', output.get_shape().as_list())
        # Apply non-linearity (if asked) and return output
        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def lstm_network(input, scope='lstm_network'):
    with tf.variable_scope(scope):
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # initial_state = lstm_cells.zero_state(batch_size,  tf.float32)

        _, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)

        z_sequence_output = states[1].h
        # print(z_sequence_output.get_shape())
        #states_concat = tf.concat([states[0].h, states[1].h], 1)
        #z_sequence_output = dense(states_concat, lstm_linear_transform_input_dim, lstm_z_sequence_dim, scope='linear_transform')

    return z_sequence_output


def g_encoder_network(x, activation='swish', scope='g_encoder_network', bn_phaze=False, b_noise=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        if use_random_noise:
            x = tf.cond(b_noise, lambda: util.add_gaussian_noise(x, 0.0, 0.1), lambda: util.add_gaussian_noise(x, 0.0, -1.0))

        g_enc_conv1 = conv(x, scope='g_enc_conv1', filter_dims=[g_encoder_input_dim, 2, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)

        g_enc_conv2 = conv(g_enc_conv1, scope='g_enc_conv2', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        g_enc_conv3 = conv(g_enc_conv2, scope='g_enc_conv3', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)

        g_enc_conv_fused1 = tf.concat([g_enc_conv2, g_enc_conv3], axis=3)
        g_enc_conv_fused1 = batch_norm_conv(g_enc_conv_fused1, b_train=bn_phaze, scope='g_enc_conv_fused1_bn')

        g_enc_conv4 = conv(g_enc_conv_fused1, scope='g_enc_conv4', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        g_enc_conv5 = conv(g_enc_conv4, scope='g_enc_conv5', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        g_enc_conv6 = conv(g_enc_conv5, scope='g_enc_conv6', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)

        g_enc_conv_fused2 = tf.concat([g_enc_conv2, g_enc_conv3, g_enc_conv4, g_enc_conv5, g_enc_conv6], axis=3)
        g_enc_conv_fused2 = batch_norm_conv(g_enc_conv_fused2, b_train=bn_phaze, scope='g_enc_conv_fused2_bn')

        g_enc_conv7 = conv(g_enc_conv_fused2, scope='g_enc_conv7', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        g_enc_conv8  = conv(g_enc_conv7, scope='g_enc_conv8', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func)
        g_enc_conv9 = conv(g_enc_conv8, scope='g_enc_conv9', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func)

        g_enc_conv_fused3 = tf.concat([g_enc_conv2, g_enc_conv3, g_enc_conv4, g_enc_conv5, g_enc_conv6, g_enc_conv7, g_enc_conv8, g_enc_conv9], axis=3)
        g_enc_conv_fused3 = batch_norm_conv(g_enc_conv_fused3, b_train=bn_phaze, scope='g_enc_conv_fused3_bn')

        g_enc_conv10= conv(g_enc_conv_fused3, scope='g_enc_conv10', filter_dims=[1, 1, 32], stride_dims=[1, 1], non_linear_fn=act_func)

        g_enc_z_local = fc(g_enc_conv10, scope='g_enc_z_fc', out_dim=g_encoder_z_local_dim, non_linear_fn=None)

        return g_enc_z_local


def g_decoder_network(x, activation='swish', scope='g_decoder_network', bn_phaze=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        # print('decoder input:', x.get_shape())
        input = tf.reshape(x, shape=[-1, 4, 4, 8])

        g_dec_conv1 = deconv(input, batch_size=batch_size, scope='g_dec_conv1', filter_dims=[3, 3, 512], stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        # print('deconv1:', g_dec_conv1.get_shape())
        g_dec_conv2 = deconv(g_dec_conv1, batch_size=batch_size, scope='g_dec_conv2', filter_dims=[3, 3, 256], stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        # print('deconv2:', g_dec_conv2.get_shape())
        g_dec_conv3 = deconv(g_dec_conv2, batch_size=batch_size, scope='g_dec_conv3', filter_dims=[3, 3, 30], stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        # print('deconv3:', g_dec_conv3.get_shape())

        g_dec_output = tf.reshape(g_dec_conv3, shape=[-1, 150, 20, 1])

        return g_dec_output


def discriminator(input_data, activation='swish', scope='discriminator', reuse=False, bn_phaze=False):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'tanh':
            act_func = tf.nn.tanh
        else:
            act_func = tf.nn.sigmoid

        '''
        input = tf.reshape(x, shape=[-1, 10, 10, 30])
        dc_conv1 = conv(input, scope='dc_conv1', filter_dims=[3, 3, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=True)
        dc_conv2 = conv(dc_conv1, scope='dc_conv2', filter_dims=[3, 3, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=True)
        dc_conv_fused1 = tf.concat([dc_conv1, dc_conv2], axis=3)

        dc_conv3 = conv(dc_conv_fused1, scope='dc_conv3', filter_dims=[3, 3, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=True)
        dc_conv4 = conv(dc_conv3, scope='dc_conv4', filter_dims=[3, 3, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=True)
        dc_conv_fused2 = tf.concat([dc_conv1, dc_conv2, dc_conv3, dc_conv4], axis=3)

        dc_conv5 = conv(dc_conv_fused2, scope='dc_conv5', filter_dims=[3, 3, 32], stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        dc_conv6 = conv(dc_conv5, scope='dc_conv6', filter_dims=[1, 1, 16], stride_dims=[1, 1], non_linear_fn=act_func)

        dc_output = fc(dc_conv6, scope='dc_fc', out_dim=1, non_linear_fn=None)
        dc_final_layer = dc_conv6
    
        return dc_final_layer, dc_output, tf.sigmoid(dc_output)
        '''

        dc_conv1 = conv(input_data, scope='dc_conv1', filter_dims=[g_encoder_input_dim, 2, 64], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        dc_conv2 = conv(dc_conv1, scope='dc_conv2', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        dc_conv3 = conv(dc_conv2, scope='dc_conv3', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)

        dc_conv_fused1 = tf.concat([dc_conv2, dc_conv3], axis=3)
        dc_conv_fused1 = batch_norm_conv(dc_conv_fused1, b_train=bn_phaze, scope='dc_conv_fused1_bn')

        dc_conv4 = conv(dc_conv_fused1, scope='dc_conv4', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        dc_conv5 = conv(dc_conv4, scope='dc_conv5', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        dc_conv6 = conv(dc_conv5, scope='dc_conv6', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)

        dc_conv_fused2 = tf.concat([dc_conv2, dc_conv3, dc_conv4, dc_conv5, dc_conv6], axis=3)
        dc_conv_fused2 = batch_norm_conv(dc_conv_fused2, b_train=bn_phaze, scope='dc_conv_fused2_bn')

        dc_conv7 = conv(dc_conv_fused2, scope='dc_conv7', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        dc_conv8 = conv(dc_conv7, scope='dc_conv8', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func)
        dc_conv9 = conv(dc_conv8, scope='dc_conv9', filter_dims=[1, 2, 32], stride_dims=[1, 1], non_linear_fn=act_func)

        dc_conv_fused3 = tf.concat([dc_conv2, dc_conv3, dc_conv4, dc_conv5, dc_conv6, dc_conv7, dc_conv8, dc_conv9], axis=3)
        dc_conv_fused3 = batch_norm_conv(dc_conv_fused3, b_train=bn_phaze, scope='dc_conv_fused3_bn')

        dc_conv10 = conv(dc_conv_fused3, scope='dc_conv10', filter_dims=[1, 1, 32], stride_dims=[1, 1], non_linear_fn=act_func)

        dc_output = fc(dc_conv10, scope='dc_fc', out_dim=1, non_linear_fn=None)

        dc_final_layer = dc_conv10

        return dc_final_layer, dc_output, tf.sigmoid(dc_output)


def batch_norm_conv(x, b_train, scope, reuse=False):
    """
    Args:
        x:           Tensor, 4D input maps (B, H, W, C)
        b_train:       Boolean, train/test mode
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def batch_norm(x, b_train, scope, reuse=False):
    """
    Args:
        x:           Tensor, 2D input maps
        b_train:       Boolean, train/test mode
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))
        #beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
        #                   name='beta', trainable=True)
        #gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
        #                    name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(b_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return normed


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        # loss = -tf.reduce_mean(x_ * tf.log(decoded + eps))
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value))))
    elif type == 'l2':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))

    return gamma * loss


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        # loss = -tf.reduce_mean(x_ * tf.log(decoded + eps))
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))

    return gamma * loss


def get_conceptual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        # loss = -tf.reduce_mean(x_ * tf.log(decoded + eps))
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))

    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        return gamma * (d_loss_real - d_loss_fake) + 1.0, d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake


def train(b_test=False):
    device = {0: '/cpu:0', 1: '/gpu:0', 2: '/gpu:1'}

    # Generate test sample
    inlier_sample, outlier_sample = util.generate_samples(150, 100000, 1000)

    bn_train = tf.placeholder(tf.bool)
    add_noise = tf.placeholder(tf.bool)

    # H: 150, W: 20, C: 1
    g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])

    lstm_input = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, input_feature_dim])
    # Z seq: LSTM sequence latent output
    z_seq = lstm_network(lstm_input, scope='LSTM')
    # z_seq = np.random.uniform(-1., 1., size=[batch_size, 32])

    with tf.device(device[2]):
        # Z enc: Encoder latent output
        z_local = g_encoder_network(g_encoder_input, activation='swish', scope='G_Encoder', bn_phaze=bn_train, b_noise=add_noise)

    with tf.device(device[0]):
        z_enc = tf.concat([z_seq, z_local], 1)

    with tf.device(device[2]):
        # Reconstructed output
        decoder_output = g_decoder_network(z_enc, activation='swish', scope='G_Decoder', bn_phaze=bn_train)

    # Discriminator output
    #   - feature real/fake: Feature matching approach. Returns last feature layer
    feature_real, d_real, d_real_output = discriminator(g_encoder_input, activation='swish', scope='Discriminator',
                                                        bn_phaze=bn_train)
    feature_fake, d_fake, d_fake_output = discriminator(decoder_output, activation='swish', scope='Discriminator',
                                                        reuse=True, bn_phaze=bn_train)

    d_fake_output = tf.squeeze(d_fake_output)
    d_real_output = tf.squeeze(d_real_output)

    # Trainable variable lists
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    g_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Encoder')
    # print('encoder vars:', g_encoder_var)
    g_decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Decoder')
    g_lstm_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LSTM')

    generator_vars = g_encoder_var + g_decoder_var + g_lstm_var

    # Joint loss term
    residual_loss = get_residual_loss(decoder_output, g_encoder_input, type='l2', gamma=1.0)
    feature_matching_loss = get_feature_matching_loss(feature_fake, feature_real, type='l2', gamma=1.0)
    # Cross Entropy
    # gan_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
    # WGAN
    gan_g_loss = tf.reduce_mean(d_fake) + 1.0
    #gan_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
    alpha = 1.0
    beta = 1.0
    gamma = 0.5
    generator_loss = alpha * residual_loss + beta * feature_matching_loss + gamma * gan_g_loss
    #discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='ce', gamma=1.0)
    discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='wgan', gamma=1.0)

    # For wgan loss.
    d_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_var]

    # training operation
    #d_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)
    f_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(feature_matching_loss, var_list=generator_vars)
    gan_g_optimzier = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(gan_g_loss, var_list=generator_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, './model/stadnet/STADNet.ckpt')
        except:
            print('Restore failed')

        if b_test == False:
            num_itr = int(len(inlier_sample)/batch_size)
            b_wgan = True

            for epoch in range(num_epoch):
                for itr in range(num_itr):
                    batch_x, batch_seq = util.get_sequence_batch(inlier_sample, width, batch_size)

                    cnn_batch_x = np.transpose(batch_seq, axes=[0, 2, 1])
                    cnn_batch_x = np.expand_dims(cnn_batch_x, axis=3)

                    _, r_loss = sess.run([g_optimizer, residual_loss],
                                         feed_dict={g_encoder_input: cnn_batch_x, lstm_input: batch_seq,
                                                    bn_train: True, add_noise: True})
                    _, f_loss = sess.run([f_optimizer, feature_matching_loss],
                                         feed_dict={g_encoder_input: cnn_batch_x, lstm_input: batch_seq,
                                                    bn_train: True, add_noise: True})
                    sess.run([gan_g_optimzier],
                             feed_dict={g_encoder_input: cnn_batch_x, lstm_input: batch_seq,
                                        bn_train: True, add_noise: True})

                    if b_wgan:
                        # wgan
                        _, _, d_loss, l_real, l_fake = sess.run([d_optimizer, d_weight_clip, discriminator_loss, loss_real, loss_fake],
                                                                feed_dict={g_encoder_input: cnn_batch_x,
                                                                           lstm_input: batch_seq,
                                                                           bn_train: True,
                                                                           add_noise: False})
                    else:
                        _, d_loss, l_real, l_fake = sess.run([d_optimizer, discriminator_loss, loss_real, loss_fake],
                                                             feed_dict={g_encoder_input: cnn_batch_x,
                                                                        lstm_input: batch_seq,
                                                                        bn_train: True,
                                                                        add_noise: True})

                    if (itr + 1) % 10 == 0:
                        print('epoch: {0}, itr: {1}, l_real: {2}, l_fake: {3}'.format(epoch, itr, l_real, l_fake))
                        print('epoch: {0}, itr: {1}, d_loss: {2}, r_loss: {3}, f_loss: {4}'.format(epoch, itr, d_loss, r_loss, f_loss))

                        try:
                            saver.save(sess, './model/stadnet/STADNet.ckpt')
                        except:
                            print('Save failed')
        else:
            for i in range(100):
                outlier_sample = outlier_sample + np.random.normal(loc=2.0, scale=0.1, size=outlier_sample.shape)
                batch_x, batch_seq = util.get_sequence_batch(outlier_sample, width, batch_size)

                cnn_batch_x = np.transpose(batch_seq, axes=[0, 2, 1])
                cnn_batch_x = np.expand_dims(cnn_batch_x, axis=3)

                d_loss, r_loss, f_loss = sess.run([d_real_output, residual_loss, feature_matching_loss],
                                                  feed_dict={g_encoder_input: cnn_batch_x, lstm_input: batch_seq,
                                                             bn_train: False, add_noise: True})

                score = 10 * r_loss
                print('outlier Anomaly Score:', score, ', d loss:', d_loss, ', r loss:', r_loss, ', f loss:', f_loss)

                batch_x, batch_seq = util.get_sequence_batch(inlier_sample, width, batch_size)

                cnn_batch_x = np.transpose(batch_seq, axes=[0, 2, 1])
                cnn_batch_x = np.expand_dims(cnn_batch_x, axis=3)

                d_loss, r_loss, f_loss = sess.run([d_real_output, residual_loss, feature_matching_loss],
                                                  feed_dict={g_encoder_input: cnn_batch_x, lstm_input: batch_seq,
                                                             bn_train: False, add_noise: True})

                score = 10 * r_loss

                print('inlier Anomaly Score:', score, ', d loss:', d_loss , ', r loss:', r_loss, ', f loss:', f_loss)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Test mode', action='store_true')
    parser.add_argument('--epoch', help='epoch count', default=1)
    parser.add_argument('--batchsize', help='batch size', default=128)
    parser.add_argument('--noise', help='Add random noise', action='store_true')

    args = parser.parse_args()

    num_epoch = args.epoch
    batch_size = args.batchsize
    use_random_noise = args.noise

    if args.train:
        train(b_test=False)
    elif args.test:
        batch_size = 1
        train(b_test=True)
    else:
        print('Please set options. --train or -- test')
