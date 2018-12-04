# Adversarial Anomaly Detection
#
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================


import tensorflow as tf
from sae import StackedAutoEncoder
import numpy as np
import util
import argparse

input_feature_dim = 150

lstm_sequence_length = 20
lstm_hidden_size_layer1 = 128
lstm_hidden_size_layer2 = 128
lstm_feature_dim = lstm_hidden_size_layer1
lstm_z_sequence_dim = 32
lstm_linear_transform_input_dim = 2 * lstm_feature_dim

g_encoder_z_local_dim = 32
g_encoder_z_dim = lstm_z_sequence_dim + g_encoder_z_local_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 128
g_encoder_layer2_dim = 96
g_encoder_layer3_dim = 64

g_decoder_output_dim = input_feature_dim
g_decoder_layer2_dim = 96
g_decoder_layer1_dim = 128

d_layer_1_dim = 128
d_layer_2_dim = 96
d_layer_3_dim = 64
d_layer_4_dim = 32

num_block_layers = 3
dense_layer_depth = 16

# Pretraining Stacked Auto Encoder
def g_encoder_pretrain(input):
    model = StackedAutoEncoder(dims=[g_encoder_layer1_dim, g_encoder_layer2_dim, g_encoder_layer3_dim], activations=['swish', 'swish', 'swish'],
                               epoch=[8000, 4000, 4000], loss_type='l1', lr=0.001, batch_size=32, print_step=200)
    model.fit(input)

    weights, biases = model.get_layers()

    tf.reset_default_graph()

    return weights, biases


def lstm_network(input, scope='lstm_network'):
    with tf.variable_scope(scope):
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

        lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

        # initial_state = lstm_cells.zero_state(batch_size,  tf.float32)

        _, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)

        # z_sequence_output = states[1].h
        # print(z_sequence_output.get_shape())
        states_concat = tf.concat([states[0].h, states[1].h], 1)
        z_sequence_output = dense(states_concat, lstm_linear_transform_input_dim, lstm_z_sequence_dim, scope='linear_transform')

    return z_sequence_output


def dense(x, n1, n2, scope='dense', initial_value=None, use_bias=True):
    with tf.variable_scope(scope):
        if initial_value is None:
            weights = tf.get_variable("weights", shape=[n1, n2], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        else:
            weights = tf.get_variable("weights", initializer=initial_value[0])
            bias = tf.get_variable("bias", initializer=initial_value[1])

        if use_bias:
            out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        else:
            out = tf.matmul(x, weights, name='matmul')

        return out


def g_encoder_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='g_encoder_network',
                      bn_phaze=False, b_noise=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        if use_random_noise:
            x = tf.cond(b_noise, lambda: util.add_gaussian_noise(x, 0.0, 0.1),
                        lambda: util.add_gaussian_noise(x, 0.0, -1.0))

        if pretrained:
            g_enc_dense_1 = act_func(dense(x, g_encoder_input_dim, g_encoder_layer1_dim, scope='g_enc_dense_1',
                                           initial_value=[weights[0], biases[0]]))
            g_enc_dense_1 = batch_norm(g_enc_dense_1, bn_phaze, scope='g_enc_dense1_bn')
            g_enc_dense_2 = act_func(dense(g_enc_dense_1, g_encoder_layer1_dim, g_encoder_layer2_dim, scope='g_enc_dense_2',
                                           initial_value=[weights[1], biases[1]]))
            g_enc_dense_2 = batch_norm(g_enc_dense_2, bn_phaze, scope='g_enc_dense2_bn')

            g_enc_dense_3 = act_func(
                dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_layer3_dim, scope='g_enc_dense_3',
                      initial_value=[weights[2], biases[2]]))
            g_enc_dense_3 = batch_norm(g_enc_dense_3, bn_phaze, scope='g_enc_dense3_bn')

            g_enc_z_local = dense(g_enc_dense_3, g_encoder_layer3_dim, g_encoder_z_local_dim, scope='g_enc_z_local')
        else:
            g_enc_dense_1 = act_func(dense(x, g_encoder_input_dim, g_encoder_layer1_dim, scope='g_enc_dense_1'))
            g_enc_dense_1 = batch_norm(g_enc_dense_1, bn_phaze, scope='g_enc_dense1_bn')

            g_enc_dense_2 = act_func(dense(g_enc_dense_1, g_encoder_layer1_dim, g_encoder_layer2_dim, scope='g_enc_dense_2'))
            g_enc_dense_2 = batch_norm(g_enc_dense_2, bn_phaze, scope='g_enc_dense2_bn')

            g_enc_dense_3 = act_func(dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_layer3_dim, scope='g_enc_dense_3'))
            g_enc_dense_3 = batch_norm(g_enc_dense_3, bn_phaze, scope='g_enc_dense3_bn')

            g_enc_z_local = dense(g_enc_dense_3, g_encoder_layer3_dim, g_encoder_z_local_dim, scope='g_enc_z_local')

        return g_enc_z_local


def g_decoder_network(x, activation='swish', scope='g_decoder_network', bn_phaze=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        g_dec_dense_2 = act_func(dense(x, g_encoder_z_dim, g_decoder_layer2_dim, scope='g_dec_dense_2'))
        g_dec_dense_2 = batch_norm(g_dec_dense_2, bn_phaze, scope='g_dec_dense2_bn')
        g_dec_dense_1 = act_func(dense(g_dec_dense_2, g_decoder_layer2_dim, g_decoder_layer1_dim, scope='g_dec_dense_1'))
        g_dec_dense_1 = batch_norm(g_dec_dense_1, bn_phaze, scope='g_dec_dense1_bn')
        g_dec_output = act_func(dense(g_dec_dense_1, g_decoder_layer1_dim, g_decoder_output_dim, scope='g_dec_output'))
        return g_dec_output


def r_encoder_network(x, activation='swish', scope='r_encoder_network', bn_phaze=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        r_enc_dense1 = act_func(dense(x, g_encoder_input_dim, g_decoder_layer1_dim, scope='r_enc_dense_1'))
        r_enc_dense1 = batch_norm(r_enc_dense1, bn_phaze, scope='r_enc_dense1_bn')
        r_enc_dense2 = act_func(dense(r_enc_dense1, g_decoder_layer1_dim, g_decoder_layer2_dim, scope='r_enc_dense_2'))
        r_enc_dense2 = batch_norm(r_enc_dense2, bn_phaze, scope='r_enc_dense2_bn')
        r_enc_output = dense(r_enc_dense2, g_decoder_layer2_dim, g_encoder_z_dim, scope='r_enc_output')
        return r_enc_output


def conv(input, scope, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu, bias=True):
    input_dims = input.get_shape().as_list()

    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        conv_weight = tf.Variable(
            tf.truncated_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))
        conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))
        map = tf.nn.conv2d(input, conv_weight, strides=[1, stride_h, stride_w, 1], padding=padding)

        if bias is True:
            map = tf.nn.bias_add(map, conv_bias)

        if non_linear_fn is not None:
            activation = non_linear_fn(map)
        else:
            activation = map

        # print(activation.get_shape().as_list())
        return activation


def fc(input, scope, out_dim, non_linear_fn=None):
    assert (type(out_dim) == int)

    with tf.variable_scope(scope):
        input_dims = input.get_shape().as_list()

        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input

        fc_weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32))

        fc_bias = tf.Variable(tf.zeros([out_dim], dtype=tf.float32))

        output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)

        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def batch_norm_conv(x, b_train, scope):
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


def add_dense_layer(layer, filter_dims, act_func=tf.nn.relu, scope='dense_layer', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = act_func(layer)

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        l = tf.concat([l, layer], 3)

    return l


def avg_pool(input, scope, filter_dims, stride_dims, padding='SAME'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.avg_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool


def add_dense_transition(layer, filter_dims, act_func=tf.nn.relu, scope='transition', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = act_func(layer)

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        # l = avg_pool(l, scope='avgpool', filter_dims=[3, 3], stride_dims=[1, 1])
    return l


def discriminator(x, activation='swish', scope='discriminator', reuse=False, bn_phaze=False, use_cnn=True):
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        if use_cnn:
            input = tf.reshape(x, shape=[-1, 5, 5, 6])

            l = conv(input, scope='dc_conv1', filter_dims=[3, 3, 64], stride_dims=[1, 1], non_linear_fn=None, bias=False)

            with tf.variable_scope('dense_block_1'):
                for i in range(num_block_layers):
                    l = add_dense_layer(l, filter_dims=[3, 3, dense_layer_depth], act_func=act_func,use_bn=False, bn_phaze=bn_phaze,
                                        scope='layer' + str(i))
                l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                         scope='dense_transition_1',
                                         use_bn=True,
                                         bn_phaze=bn_phaze)

            with tf.variable_scope('dense_block_2'):
                for i in range(num_block_layers):
                    l = add_dense_layer(l, filter_dims=[3, 3, dense_layer_depth], act_func=act_func, use_bn=False, bn_phaze=bn_phaze,
                                        scope='layer' + str(i))
                l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                         scope='dense_transition_1',
                                         use_bn=True,
                                         bn_phaze=bn_phaze)

            with tf.variable_scope('dense_block_3'):
                for i in range(num_block_layers):
                    l = add_dense_layer(l, filter_dims=[3, 3, dense_layer_depth], act_func=act_func, use_bn=False,
                                        bn_phaze=bn_phaze, scope='layer' + str(i))
                l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func, scope='dense_transition_1',
                                         bn_phaze=bn_phaze)

            with tf.variable_scope('dense_block_4'):
                for i in range(num_block_layers):
                    l = add_dense_layer(l, filter_dims=[3, 3, dense_layer_depth], act_func=act_func, use_bn=False,
                                        bn_phaze=bn_phaze, scope='layer' + str(i))
                l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func, scope='dense_transition_1',
                                         bn_phaze=bn_phaze)

            with tf.variable_scope('dense_block_5'):
                for i in range(num_block_layers):
                    l = add_dense_layer(l, filter_dims=[3, 3, dense_layer_depth], act_func=act_func, use_bn=False,
                                        bn_phaze=bn_phaze, scope='layer' + str(i))
                last_dense_layer = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                                        scope='dense_transition_1', use_bn=True, bn_phaze=bn_phaze)

                #dc_final_layer = batch_norm_conv(last_dense_layer, b_train=bn_phaze, scope='last_dense_layer')
                dc_final_layer = last_dense_layer

                dc_output = fc(dc_final_layer, scope='g_enc_z_fc', out_dim=1, non_linear_fn=None)
        else:
            dc_den1 = act_func(dense(x, input_feature_dim, d_layer_1_dim, scope='dc_dense_1'))
            #dc_den1 = batch_norm(dc_den1, bn_phaze, scope='dc_den1_bn')
            dc_den2 = act_func(dense(dc_den1, d_layer_1_dim, d_layer_2_dim, scope='dc_dense_2'))
            #dc_den2 = batch_norm(dc_den2, bn_phaze, scope='dc_den2_bn')
            dc_den3 = act_func(dense(dc_den2, d_layer_2_dim, d_layer_3_dim, scope='dc_dense_3'))
            #dc_den3 = batch_norm(dc_den3, bn_phaze, scope='dc_den3_bn')
            dc_den4 = act_func(dense(dc_den3, d_layer_3_dim, d_layer_4_dim, scope='dc_dense_4'))
            #dc_den4 = batch_norm(dc_den4, bn_phaze, scope='dc_den4_bn')
            dc_output = dense(dc_den4, d_layer_4_dim, 1, scope='dc_output')
            dc_final_layer = dc_den4

        return dc_final_layer, dc_output, tf.sigmoid(dc_output)


def batch_norm(x, b_train, scope, reuse=False):
    with tf.variable_scope(scope,  reuse=tf.AUTO_REUSE):
        n_out = x.get_shape().as_list()[-1]

        beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
        gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))

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
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_sum(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))

    return gamma * loss


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
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
        loss = tf.reduce_sum(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))

    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake


def train(pretrain=True, b_test=False):
    device = {0: '/cpu:0', 1: '/gpu:0', 2: '/gpu:1'}
    b_wgan = False

    # Generate test sample
    inlier_sample, outlier_sample = util.generate_samples(150, 100000, 100)

    with tf.device(device[2]):
        # Pretraining Stacked Auto Encoder
        if pretrain:
            stacked_auto_encoder_weights, stacked_auto_encoder_biases = g_encoder_pretrain(inlier_sample)
        else:
            stacked_auto_encoder_weights = None
            stacked_auto_encoder_biases = None

    add_noise = tf.placeholder(tf.bool)
    bn_train = tf.placeholder(tf.bool)
    g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, input_feature_dim])

    with tf.device(device[2]):
        # Z local: Encoder latent output
        z_local = g_encoder_network(g_encoder_input, pretrained=pretrain,
                                    weights=stacked_auto_encoder_weights, biases=stacked_auto_encoder_biases,
                                    activation='swish', scope='G_Encoder', bn_phaze=bn_train, b_noise=add_noise)

    lstm_input = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, input_feature_dim])
    # Z seq: LSTM sequence latent output
    z_seq = lstm_network(lstm_input, scope='LSTM')
    # z_seq = np.random.uniform(-1., 1., size=[batch_size, 32])

    with tf.device(device[0]):
        z_enc = tf.concat([z_seq, z_local], 1)


    # Reconstructed output
    decoder_output = g_decoder_network(z_enc, activation='swish', scope='G_Decoder', bn_phaze=bn_train)

    # Reencoding reconstucted output
    z_renc = r_encoder_network(decoder_output, activation='swish', scope='R_Encoder', bn_phaze=bn_train)

    with tf.device(device[2]):
        # Discriminator output
        #   - feature real/fake: Feature matching approach. Returns last feature layer
        feature_real, d_real, d_real_output = discriminator(g_encoder_input, activation='swish', scope='Discriminator', bn_phaze=bn_train)
        feature_fake, d_fake, d_fake_output = discriminator(decoder_output, activation='swish', scope='Discriminator', reuse=True, bn_phaze=bn_train)

        d_real_output = tf.squeeze(d_real_output)
        d_fake_output = tf.squeeze(d_fake_output)

    # Trainable variable lists
    with tf.device(device[2]):
        d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    r_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='R_Encoder')
    g_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Encoder')

    g_decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Decoder')
    lstm_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LSTM')
    generator_vars = lstm_var + g_encoder_var + g_decoder_var
    conceptual_vars = g_decoder_var + r_encoder_var

    # Joint loss term
    residual_loss = get_residual_loss(decoder_output, g_encoder_input, type='l1', gamma=1.0)
    feature_matching_loss = get_feature_matching_loss(feature_fake, feature_real, type='l2', gamma=1.0)

    if b_wgan:
        # WGAN
        gan_g_loss = -tf.reduce_mean(d_fake)
        discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='wgan', gamma=1.0)
        d_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_var]
    else:
        # Cross Entropy
        gan_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='ce', gamma=1.0)

    conceptual_loss = get_conceptual_loss(z_renc, z_enc, type='l2', gamma=1.0)

    with tf.device(device[2]):
        # training operation
        #d_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)

    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)
    f_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(feature_matching_loss, var_list=generator_vars)
    gan_g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(gan_g_loss, var_list=generator_vars)
    r_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(conceptual_loss, var_list=[conceptual_vars, generator_vars])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, './model/seadnet/SEADNet.ckpt')
        except:
            print('Start New Training. Wait...')

        if b_test == False:
            num_itr = int(len(inlier_sample)/batch_size)

            for epoch in range(num_epoch):
                for itr in range(num_itr):
                    batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, batch_size)

                    if b_wgan:
                        _, _, d_loss, l_real, l_fake = sess.run(
                            [d_optimizer, d_weight_clip, discriminator_loss, loss_real, loss_fake],
                            feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True,
                                       add_noise: True})
                    else:
                        _, d_loss, l_real, l_fake = sess.run([d_optimizer, discriminator_loss, loss_real, loss_fake],
                                                             feed_dict={g_encoder_input: batch_x,
                                                                        lstm_input: batch_seq, bn_train: True,
                                                                        add_noise: True})

                        _, f_loss = sess.run([f_optimizer, feature_matching_loss],
                                             feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True,
                                                        add_noise: True})

                    _, g_loss = sess.run([gan_g_optimizer, gan_g_loss],
                                         feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True,
                                         add_noise: True})

                    _, r_loss = sess.run([g_optimizer, residual_loss],
                                         feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True,
                                                    add_noise: True})

                    _, c_loss = sess.run([r_optimizer, conceptual_loss],
                                         feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True,
                                                    add_noise: True})

                    if (itr + 1) % 200 == 0:
                        print('epoch: {0}, itr: {1}, l_real: {2}, l_fake: {3}'.format(epoch, itr, l_real, l_fake))
                        print('epoch: {0}, itr: {1}, d_loss: {2}, g_loss: {3},  r_loss: {4}, c_loss: {5}'.format(
                            epoch, itr, d_loss, g_loss, r_loss, c_loss))
                        try:
                            saver.save(sess, './model/seadnet/SEADNet.ckpt')
                        except:
                            print('Save failed')
        else:
            for i in range(100):
                batch_x, batch_seq = util.get_sequence_batch(outlier_sample, lstm_sequence_length, 1)

                # batch_x = np.ones_like(batch_x)
                # batch_x = [np.random.binomial(1, 0.5, 150)]

                d_real_loss, d_fake_loss, r_loss, c_loss = sess.run(
                    [d_real_output, d_fake_output, residual_loss, conceptual_loss],
                    feed_dict={g_encoder_input: batch_x,
                               lstm_input: batch_seq, bn_train: False, add_noise: True})

                score = 10 * (r_loss + c_loss)

                if b_wgan:
                    print('Outlier Anomaly Score:', score, ',d fake loss:', d_fake_loss, ',d real loss:',
                          d_real_loss, ', r loss:', r_loss, ', c loss:', c_loss)
                else:
                    print('Outlier Anomaly Score:', score, ',d fake loss:', d_fake_loss, ',d real loss:',
                          d_real_loss, ', r loss:', r_loss, ', c loss:', c_loss)

                batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, 1)

                d_real_loss, d_fake_loss, r_loss, f_loss, c_loss = sess.run(
                    [d_real_output, d_fake_output, residual_loss, feature_matching_loss, conceptual_loss],
                    feed_dict={g_encoder_input: batch_x,
                               lstm_input: batch_seq, bn_train: False, add_noise: True})

                score = 10 * (r_loss + c_loss)

                if b_wgan:
                    print('Inlier Anomaly Score:', score, ',d fake loss:', d_fake_loss, ',d real loss:',
                          d_real_loss, ', r loss:', r_loss, ', c loss:', c_loss)
                else:
                    print('Inlier Anomaly Score:', score, ',d fake loss:', d_fake_loss, ',d real loss:',
                          d_real_loss, ', r loss:', r_loss, ', c loss:', c_loss)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Test mode', action='store_true')
    parser.add_argument('--noise', help='Add random noise', action='store_true')
    parser.add_argument('--sae', help='Pretrain encoder', action='store_true')
    parser.add_argument('--epoch', type=int, help='epoch count', default=1)
    parser.add_argument('--batchsize', type=int, help='batch size', default=128)

    args = parser.parse_args()

    num_epoch = args.epoch
    batch_size = args.batchsize
    use_random_noise = args.noise

    if args.train:
        train(pretrain=args.sae, b_test=False)
    elif args.test:
        train(pretrain=False, b_test=True)
    else:
        print('Please set options. --train or -- test, for training with sae use --train --sae')
