# Adversarial Anomaly Detection
#
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
#
# ==============================================================================

import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import util
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_feature_dim = 150

width = 20
height = input_feature_dim
num_block_layers = 3
dense_layer_depth = 16

g_encoder_z_local_dim = 64
g_encoder_z_dim = g_encoder_z_local_dim
g_encoder_input_dim = input_feature_dim
g_encoder_layer1_dim = 128
g_encoder_layer2_dim = 64

g_sequence_length = 20

g_decoder_output_dim = input_feature_dim


def dense(x, n1, n2, scope='dense', initial_value=None):
    with tf.variable_scope(scope):
        if initial_value is None:
            weights = tf.get_variable("weights", shape=[n1, n2],
                                      initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
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


def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    _, input_h, input_w, num_channels_in = input_dims
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    if padding == 'SAME':
        out_h = input_h * stride_h
    elif padding == 'VALID':
        out_h = (input_h - 1) * stride_h + filter_h

    if padding == 'SAME':
        out_w = input_w * stride_w
    elif padding == 'VALID':
        out_w = (input_w - 1) * stride_w + filter_w

    return [batch_size, out_h, out_w, num_channels_out]


def deconv(input_data, b_size, scope, filter_dims, stride_dims, padding='SAME', non_linear_fn=tf.nn.relu):
    input_dims = input_data.get_shape().as_list()
    # print(scope, 'in', input_dims)
    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(stride_dims) == 2)  # stride height and width

    input_dims = [b_size, input_dims[1], input_dims[2], input_dims[3]]
    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    with tf.variable_scope(scope):
        deconv_weight = tf.Variable(
            tf.random_normal([filter_h, filter_w, num_channels_out, num_channels_in], stddev=0.1, dtype=tf.float32))

        deconv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))

        map = tf.nn.conv2d_transpose(input_data, deconv_weight, output_dims, strides=[1, stride_h, stride_w, 1],
                                     padding=padding)

        map = tf.nn.bias_add(map, deconv_bias)

        activation = non_linear_fn(map)

        # print(scope, 'out', activation.get_shape().as_list())
        return activation


def avg_pool(input_data, scope, filter_dims, stride_dims, padding='SAME'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.avg_pool(input_data, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool


def max_pool(input_data, scope, filter_dims, stride_dims, padding='SAME'):
    assert (len(filter_dims) == 2)  # filter height and width
    assert (len(stride_dims) == 2)  # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(scope):
        pool = tf.nn.max_pool(input_data, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1],
                              padding=padding)

        return pool


def fc(input_data, scope, out_dim, non_linear_fn=None):
    assert (type(out_dim) == int)

    with tf.variable_scope(scope):
        input_dims = input_data.get_shape().as_list()
        # print(scope, 'in', input_dims)

        if len(input_dims) == 4:
            _, input_h, input_w, num_channels = input_dims
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input_data, [-1, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input_data

        fc_weight = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32))

        fc_bias = tf.Variable(tf.zeros([out_dim], dtype=tf.float32))

        output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)

        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation


def add_dense_layer(layer, filter_dims, act_func=tf.nn.relu, scope='dense_layer', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = act_func(layer)

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
        l = tf.concat([l, layer], 3)

    return l


def add_dense_transition(layer, filter_dims, act_func=tf.nn.relu, scope='transition', use_bn=True, bn_phaze=False):
    with tf.variable_scope(scope):
        l = act_func(layer)

        if use_bn:
            l = batch_norm_conv(l, b_train=bn_phaze, scope='bn')

        l = conv(l, scope='conv', filter_dims=filter_dims, stride_dims=[1, 1], non_linear_fn=act_func, bias=False)
    return l


def g_encoder_network(x, activation='swish', scope='g_encoder_network', bn_phaze=False):
    with tf.variable_scope(scope):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        else:
            act_func = tf.nn.sigmoid

        if use_random_noise:
            #x = util.add_gaussian_noise(x, 0.0, 0.1)
            x = util.add_uniform_noise(x, 0.0, 0.1)

        l = conv(x, scope='g_enc_conv1', filter_dims=[g_encoder_input_dim, 2, 64], stride_dims=[1, 1],
                 non_linear_fn=None, bias=False)

        with tf.variable_scope('dense_block_1'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, use_bn=False, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_2'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, use_bn=False, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                     scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_3'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, 32], act_func=act_func, use_bn=False, bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, 32], act_func=act_func, scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_4'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, 32], act_func=act_func, use_bn=False, bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, 32], act_func=act_func, scope='dense_transition_1', bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_final'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, bn_phaze=bn_phaze,
                                    scope='layer' + str(i))
            last_dense_layer = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                                    scope='dense_transition_1', bn_phaze=bn_phaze)

        last_dense_layer = batch_norm_conv(last_dense_layer, b_train=bn_phaze, scope='last_dense_layer')

        g_enc_z_local = fc(last_dense_layer, scope='g_enc_z_fc', out_dim=g_encoder_z_local_dim, non_linear_fn=None)

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

        g_dec_conv1 = deconv(input, b_size=batch_size, scope='g_dec_conv1', filter_dims=[3, 3, 512],
                             stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        # print('deconv1:', g_dec_conv1.get_shape())
        g_dec_conv2 = deconv(g_dec_conv1, b_size=batch_size, scope='g_dec_conv2', filter_dims=[3, 3, 256],
                             stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
        # print('deconv2:', g_dec_conv2.get_shape())
        g_dec_conv3 = deconv(g_dec_conv2, b_size=batch_size, scope='g_dec_conv3', filter_dims=[3, 3, 30],
                             stride_dims=[1, 1], padding='VALID', non_linear_fn=act_func)
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

        l = conv(input_data, scope='dc_conv1', filter_dims=[g_encoder_input_dim, 2, 64], stride_dims=[1, 1],
                 non_linear_fn=None, bias=False)

        with tf.variable_scope('dense_block_1'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, use_bn=False,
                                    bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                     scope='dense_transition_1',
                                     bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_2'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, use_bn=False,
                                    bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                     scope='dense_transition_1',
                                     bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_3'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, 32], act_func=act_func, use_bn=False, bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, 32], act_func=act_func, scope='dense_transition_1',
                                     bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_4'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, 32], act_func=act_func, use_bn=False, bn_phaze=bn_phaze, scope='layer' + str(i))
            l = add_dense_transition(l, filter_dims=[1, 1, 32], act_func=act_func, scope='dense_transition_1',
                                     bn_phaze=bn_phaze)

        with tf.variable_scope('dense_block_5'):
            for i in range(num_block_layers):
                l = add_dense_layer(l, filter_dims=[1, 2, dense_layer_depth], act_func=act_func, use_bn=False,
                                    bn_phaze=bn_phaze, scope='layer' + str(i))
            last_dense_layer = add_dense_transition(l, filter_dims=[1, 1, dense_layer_depth], act_func=act_func,
                                                    scope='dense_transition_1', bn_phaze=bn_phaze)

            # dc_final_layer = batch_norm_conv(last_dense_layer, b_train=bn_phaze, scope='last_dense_layer')
            dc_final_layer = last_dense_layer

            dc_output = fc(dc_final_layer, scope='g_enc_z_fc', out_dim=1, non_linear_fn=None)

        return dc_final_layer, dc_output, tf.sigmoid(dc_output)


def batch_norm_conv(x, b_train, scope, reuse=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

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
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    return gamma * loss


def get_conceptual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

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


def train():
    b_wgan = True

    bn_train = tf.placeholder(tf.bool)

    # H: 150, W: 20, C: 1
    g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])

    z_rand = tf.random_uniform(shape=[batch_size, g_encoder_z_local_dim], minval=0.0, maxval=1.0)

    with tf.device(gpus[2 % num_gpus]):
        # Z enc: Encoder latent output
        z_local = g_encoder_network(g_encoder_input, activation='swish', scope='G_Encoder', bn_phaze=bn_train)

    with tf.device(cpu):
        z_enc = tf.concat([z_rand, z_local], 1)

    with tf.device(gpus[2 % num_gpus]):
        # Reconstructed output
        decoder_output = g_decoder_network(z_enc, activation='swish', scope='G_Decoder', bn_phaze=bn_train)

    with tf.device(gpus[3 % num_gpus]):
        # Discriminator output
        #   - feature real/fake: Feature matching approach. Returns last feature layer
        feature_real, d_real, d_real_output = discriminator(g_encoder_input, activation='swish', scope='Discriminator',
                                                            bn_phaze=bn_train)
        feature_fake, d_fake, d_fake_output = discriminator(decoder_output, activation='swish', scope='Discriminator',
                                                            reuse=True, bn_phaze=bn_train)

    # Trainable variable lists
    d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
    g_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Encoder')
    g_decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Decoder')

    generator_vars = g_encoder_var + g_decoder_var

    # Joint loss term
    residual_loss = get_residual_loss(decoder_output, g_encoder_input, type='l1', gamma=1.0)
    feature_matching_loss = get_feature_matching_loss(feature_fake, feature_real, type='l2', gamma=1.0)

    if b_wgan:
        # WGAN
        eps = tf.random_uniform([batch_size, 1], minval=0.0, maxval=1.0)
        gp_encoder_input = tf.reshape(g_encoder_input, [batch_size, -1])
        gp_decoder_output = tf.reshape(decoder_output, [batch_size, -1])
        gp_input = eps * gp_encoder_input + (1.0 - eps) * gp_decoder_output
        gp_input = tf.reshape(gp_input, [batch_size, height, width, 1])
        _, gp_output, _ = discriminator(gp_input, activation='swish', scope='Discriminator', reuse=True,
                                        bn_phaze=bn_train)
        gp_grad = tf.gradients(gp_output, [gp_input])[0]
        gp_grad_norm = tf.sqrt(tf.reduce_mean((gp_grad) ** 2, axis=1))
        gp_grad_pen = 10 * tf.reduce_mean((gp_grad_norm - 1) ** 2)
        gan_g_loss = -tf.reduce_mean(d_fake)
        discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='wgan', gamma=1.0)
        discriminator_loss = discriminator_loss + gp_grad_pen
        d_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_var]
    else:
        # Cross Entropy
        gan_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        discriminator_loss, loss_real, loss_fake = get_discriminator_loss(d_real, d_fake, type='ce', gamma=1.0)

    # training operation
    # d_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)
    f_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(feature_matching_loss, var_list=generator_vars)
    gan_g_optimzier = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(gan_g_loss, var_list=generator_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, './model/sgadnet/SGADNet.ckpt')
        except:
            print('Start New Training. Wait...')

        num_itr = int(len(inlier_sample) / batch_size)

        for epoch in range(num_epoch):
            for itr in range(num_itr):
                batch_x, batch_seq = util.get_sequence_batch(inlier_sample, width, batch_size)

                cnn_batch_x = np.transpose(batch_seq, axes=[0, 2, 1])
                cnn_batch_x = np.expand_dims(cnn_batch_x, axis=3)

                _, r_loss = sess.run([g_optimizer, residual_loss],
                                     feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})

                _, g_loss = sess.run([gan_g_optimzier, gan_g_loss],
                                     feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})

                if b_wgan:
                    _, _, d_loss, l_real, l_fake = sess.run(
                        [d_optimizer, d_weight_clip, discriminator_loss, loss_real, loss_fake],
                        feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})
                else:
                    _, d_loss, l_real, l_fake = sess.run([d_optimizer, discriminator_loss, loss_real, loss_fake],
                                                         feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})
                _, f_loss = sess.run([f_optimizer, feature_matching_loss],
                                     feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})

                if (itr + 1) % 10 == 0:
                    print('epoch: {0}, itr: {1}, l_real: {2}, l_fake: {3}'.format(epoch, itr, l_real, l_fake))
                    print('epoch: {0}, itr: {1}, d_loss: {2}, g_loss: {3}, r_loss: {4}'.format(
                        epoch, itr, d_loss, g_loss, r_loss))

                    try:
                        saver.save(sess, './model/sgadnet/SGADNet.ckpt')
                    except:
                        print('Save failed')


def test(input_seq, num_itr, seed=1):
    tf.reset_default_graph()

    bn_train = tf.placeholder(tf.bool)

    # H: 150, W: 20, C: 1
    g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])

    z_rand = tf.random_uniform(shape=[batch_size, g_encoder_z_local_dim], minval=0.0, maxval=1.0, seed=seed)

    with tf.device(gpus[2 % num_gpus]):
        # Z enc: Encoder latent output
        z_local = g_encoder_network(g_encoder_input, activation='swish', scope='G_Encoder', bn_phaze=bn_train)

    with tf.device(cpu):
        z_enc = tf.concat([z_rand, z_local], 1)

    with tf.device(gpus[2 % num_gpus]):
        # Reconstructed output
        decoder_output = g_decoder_network(z_enc, activation='swish', scope='G_Decoder', bn_phaze=bn_train)

    # Trainable variable lists
    g_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Encoder')
    g_decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Decoder')
    generator_vars = g_encoder_var + g_decoder_var

    # Joint loss term
    residual_loss = get_residual_loss(decoder_output, g_encoder_input, type='l1', gamma=1.0)

    # training operation
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, './model/sgadnet/SGADNet.ckpt')
        except:
            print('Fail to load')
            return

        r_loss_sum = []

        cnn_batch_x = np.transpose(input_seq, axes=[0, 2, 1])
        cnn_batch_x = np.expand_dims(cnn_batch_x, axis=3)

        for itr in range(num_itr):
            _ = sess.run([g_optimizer], feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})
            r_loss = sess.run([residual_loss], feed_dict={g_encoder_input: cnn_batch_x, bn_train: True})
            r_loss_sum.append(r_loss)

        r_loss_mean = np.mean(r_loss_sum)

        return r_loss_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--test', help='Test mode', action='store_true')
    parser.add_argument('--epoch', type=int, help='epoch count', default=1)
    parser.add_argument('--batchsize', type=int, help='batch size', default=32) # 128 batch makes OOM
    parser.add_argument('--noise', help='Add random noise', action='store_true')

    args = parser.parse_args()

    num_epoch = args.epoch
    batch_size = args.batchsize
    use_random_noise = args.noise

    # Generate test sample
    inlier_sample, outlier_sample = util.generate_samples(150, 100000, 100)

    cpu = '/device:CPU:0'
    gpus = [dev.name for dev in device_lib.list_local_devices() if dev.device_type == 'GPU']
    num_gpus = len(gpus)

    if num_gpus == 0:  # No cuda supported gpu
        num_gpus = 1
        gpus = [cpu]

    if args.train:
        train()
    elif args.test:
        batch_size = 1 # deconv limitation.
        num_seed = 8
        num_itr = 5
        seed = 0

        score_list = []

        for i in range(10):
            _, data_seq = util.get_sequence_batch(inlier_sample, g_sequence_length, 1)
            scale = len(data_seq[0]) * len(data_seq[0][0])

            for j in range(num_seed):
                recon_loss = test(data_seq, num_itr, seed)
                score_list.append(recon_loss)
                seed = seed + 1

            score = scale * np.mean(score_list)
            print('Test {0}, Inlier Anomaly Score: {1}'.format(i, score))

            score_list = []
            seed = 0

            # Noise injection [1, 20, 150]
            data_seq[0][5] = data_seq[0][5] + np.random.normal(loc=0.0, scale=1.0, size=data_seq[0][0].shape)

            for j in range(num_seed):
                recon_loss = test(data_seq, num_itr, seed)
                score_list.append(recon_loss)
                seed = seed + 1

            score = scale * np.mean(score_list)

            print('Test {0}, Outlier Anomaly Score: {1}'.format(i, score))
            print()
    else:
        print('Please set options. --train or -- test')
