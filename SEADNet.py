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

input_feature_dim = 150
batch_size = 128

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
g_encoder_layer2_dim = 64

g_decoder_output_dim = input_feature_dim
g_decoder_layer2_dim = 96
g_decoder_layer1_dim = 128

d_layer_1_dim = 128
d_layer_2_dim = 64

# Pretraining Stacked Auto Encoder
def g_encoder_pretrain(input):
	model = StackedAutoEncoder(dims=[g_encoder_layer1_dim, g_encoder_layer2_dim], activations=['swish', 'swish'],
								epoch=[10000, 10000], loss_type='l1', lr=0.001, batch_size=32, print_step=200)
	model.fit(input)

	weights, biases = model.get_layers()

	tf.reset_default_graph()

	return weights, biases


def lstm_network(input, scope='lstm_network'):
	with tf.variable_scope(scope):
		lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer1, forget_bias=1.0)
		lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size_layer2, forget_bias=1.0)

		lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)

		# initial_state = lstm_cells.zero_state(batch_size, tf.float32)

		_, states = tf.nn.dynamic_rnn(lstm_cells, input, dtype=tf.float32, initial_state=None)

		# z_sequence_output = states[1].h
		# print(z_sequence_output.get_shape())
		states_concat = tf.concat([states[0].h, states[1].h], 1)
		z_sequence_output = dense(states_concat, lstm_linear_transform_input_dim, lstm_z_sequence_dim, scope='linear_transform')

		return z_sequence_output


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


def g_encoder_network(x, pretrained=False, weights=None, biases=None, activation='swish', scope='g_encoder_network', bn_phaze=False):
	with tf.variable_scope(scope):
		if activation == 'swish':
			act_func = util.swish
		elif activation == 'relu':
			act_func = tf.nn.relu
		else:
			act_func = tf.nn.sigmoid

		if pretrained:
			g_enc_dense_1 = act_func(dense(x, g_encoder_input_dim, g_encoder_layer1_dim, scope='g_enc_dense_1',
			initial_value=[weights[0], biases[0]]))
			g_enc_dense_1 = batch_norm(g_enc_dense_1, bn_phaze, scope='g_enc_dense1_bn')
			g_enc_dense_2 = act_func(dense(g_enc_dense_1, g_encoder_layer1_dim, g_encoder_layer2_dim, scope='g_enc_dense_2',
			initial_value=[weights[1], biases[1]]))
			g_enc_dense_2 = batch_norm(g_enc_dense_2, bn_phaze, scope='g_enc_dense2_bn')
			g_enc_z_local = dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_z_local_dim, scope='g_enc_z_local')
		else:
			g_enc_dense_1 = act_func(dense(x, g_encoder_input_dim, g_encoder_layer1_dim, scope='g_enc_dense_1'))
			g_enc_dense_1 = batch_norm(g_enc_dense_1, bn_phaze, scope='g_enc_dense1_bn')
			g_enc_dense_2 = act_func(dense(g_enc_dense_1, g_encoder_layer1_dim, g_encoder_layer2_dim, scope='g_enc_dense_2'))
			g_enc_dense_2 = batch_norm(g_enc_dense_2, bn_phaze, scope='g_enc_dense2_bn')
			g_enc_z_local = dense(g_enc_dense_2, g_encoder_layer2_dim, g_encoder_z_local_dim, scope='g_enc_z_local')

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


def discriminator(x, activation='swish', scope='discriminator', reuse=False, bn_phaze=False):
	with tf.variable_scope(scope):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		if activation == 'swish':
			act_func = util.swish
		elif activation == 'relu':
			act_func = tf.nn.relu
		else:
			act_func = tf.nn.sigmoid

		dc_den1 = act_func(dense(x, input_feature_dim, d_layer_1_dim, scope='dc_dense_1'))
		#dc_den1 = batch_norm(dc_den1, bn_phaze, scope='dc_den1_bn')
		dc_den2 = act_func(dense(dc_den1, d_layer_1_dim, d_layer_2_dim, scope='dc_dense_2'))
		#dc_den2 = batch_norm(dc_den2, bn_phaze, scope='dc_den2_bn')
		dc_output = dense(dc_den2, d_layer_2_dim, 1, scope='dc_output')

		return dc_den2, dc_output, tf.sigmoid(dc_output)


def batch_norm(x, b_train, scope, reuse=False):
"""
Args:
x: Tensor, 2D input maps
b_train: Boolean, train/test mode
Return:
normed: batch-normalized maps
"""
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		n_out = x.get_shape().as_list()[-1]

		beta = tf.get_variable('beta', initializer=tf.constant(0.0, shape=[n_out]))
		gamma = tf.get_variable('gamma', initializer=tf.constant(1.0, shape=[n_out]))
		#beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
		# name='beta', trainable=True)
		#gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
		# name='gamma', trainable=True)

		batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)


		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
	
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)


		mean, var = tf.cond(b_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
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
		return gamma * (d_loss_fake - d_loss_real)
	elif type == 'ce':
		# cross entropy
		d_loss_real = tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
		d_loss_fake = tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
		return gamma * (d_loss_fake + d_loss_real)


def train(pretrain=True, b_test=False):
	# Generate test sample
	inlier_sample, outlier_sample = util.generate_samples(150, 100000, 100)

	# Pretraining Stacked Auto Encoder
	if pretrain:
		stacked_auto_encoder_weights, stacked_auto_encoder_biases = g_encoder_pretrain(inlier_sample)
	else:
		stacked_auto_encoder_weights = None
		stacked_auto_encoder_biases = None

	bn_train = tf.placeholder(tf.bool)

	lstm_input = tf.placeholder(dtype=tf.float32, shape=[None, lstm_sequence_length, input_feature_dim])
	#lstm_linear_transform_weight = tf.Variable(
	# tf.truncated_normal([lstm_linear_transform_input_dim, lstm_z_sequence_dim], stddev=0.1, dtype=tf.float32))
	#lstm_linear_transform_bias = tf.Variable(tf.zeros([lstm_z_sequence_dim], dtype=tf.float32))

	g_encoder_input = tf.placeholder(dtype=tf.float32, shape=[None, input_feature_dim])
	d_input = tf.placeholder(dtype=tf.float32, shape=[None, input_feature_dim])

	# Z local: Encoder latent output
	z_local = g_encoder_network(g_encoder_input, pretrained=pretrain,
	weights=stacked_auto_encoder_weights, biases=stacked_auto_encoder_biases,
	activation='swish', scope='G_Encoder', bn_phaze=bn_train)

	# Z seq: LSTM sequence latent output
	z_seq = lstm_network(lstm_input, scope='LSTM')

	z_enc = tf.concat([z_seq, z_local], 1)

	# Reconstructed output
	decoder_output = g_decoder_network(z_enc, activation='swish', scope='G_Decoder', bn_phaze=bn_train)

	# Reencoding reconstucted output
	z_renc = r_encoder_network(decoder_output, activation='swish', scope='R_Encoder', bn_phaze=bn_train)

	# Discriminator output
	# - feature real/fake: Feature matching approach. Returns last feature layer
	feature_real, d_real, d_real_output = discriminator(d_input, activation='relu', scope='Discriminator', bn_phaze=bn_train)
	feature_fake, d_fake, d_fake_output = discriminator(decoder_output, activation='relu', scope='Discriminator', reuse=True, bn_phaze=bn_train)

	d_real_output = tf.squeeze(d_real_output)
	d_fake_output = tf.squeeze(d_fake_output)

	# Trainable variable lists
	d_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
	r_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='R_Encoder')
	g_encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Encoder')
	# print('encoder vars:', g_encoder_var)
	g_decoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Decoder')
	lstm_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LSTM')

	generator_vars = lstm_var + g_encoder_var + g_decoder_var
	conceptual_vars = g_decoder_var + r_encoder_var

	# Joint loss term
	residual_loss = get_residual_loss(decoder_output, g_encoder_input, type='l1', gamma=1.0)
	feature_matching_loss = get_feature_matching_loss(feature_fake, feature_real, type='l2', gamma=1.0)
	alpha = 0.5
	generator_loss = alpha * residual_loss + (1-alpha) * feature_matching_loss
	conceptual_loss = get_conceptual_loss(z_renc, z_enc, type='l2', gamma=1.0)
	discriminator_loss = get_discriminator_loss(d_real, d_fake, type='ce', gamma=1.0)
	# discriminator_loss = get_discriminator_loss(d_real, d_fake, type='wgan', gamma=1.0)

	# For wgan loss.
	d_weight_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_var]

	# training operation
	d_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
	# d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(discriminator_loss, var_list=d_var)
	# g_res_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(residual_loss, var_list=generator_vars)
	# g_feature_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(feature_matching_loss,
	# var_list=generator_vars)
	g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(generator_loss, var_list=generator_vars)
	r_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(conceptual_loss, var_list=[conceptual_vars, generator_vars])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		try:
			saver = tf.train.Saver()
			saver.restore(sess, './model/SE_ADNet_L.ckpt')
		except:
			print('Restore failed')

		if b_test == False:
			num_itr = int(len(inlier_sample)/batch_size)
			num_epoch = 4
			early_stop = False
			f_loss_list = []

			for epoch in range(num_epoch):
				for itr in range(num_itr):
					batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, batch_size)

					# wgan
					#_, _, d_loss = sess.run([d_optimizer, d_weight_clip, discriminator_loss],
					# feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True})

					# gan cross entropy. 2(discriminator):1(generator) training.
					sess.run([d_optimizer, discriminator_loss], feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True})

					batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, batch_size)

					_, d_loss = sess.run([d_optimizer, discriminator_loss], feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq, bn_train:True})

					_, r_loss, f_loss = sess.run([g_optimizer, residual_loss, feature_matching_loss], feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True})

					# Test.
					#_, r_loss = sess.run([g_optimizer, residual_loss],
					# feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq})

					_, c_loss = sess.run([r_optimizer, conceptual_loss], feed_dict={g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: True})

					if itr % 200 == 0:
						print('epoch: {0}, itr: {1}, d_loss: {2}, r_loss: {3}, c_loss: {4}, f_loss: {5}'.format(epoch, itr, d_loss, r_loss, c_loss, f_loss))
						f_loss_list.append(f_loss)
						if len(f_loss_list) > 10:
							if sum(f_loss_list[-5:])/5 < 0.002:
								early_stop = False

					if early_stop:
						break
				if early_stop:
					break

			for i in range(10):
				batch_x, batch_seq = util.get_sequence_batch(outlier_sample, lstm_sequence_length, 1)

				d_loss, r_loss, f_loss, c_loss = sess.run([d_fake_output, residual_loss, feature_matching_loss, conceptual_loss],
				feed_dict={d_input: batch_x, g_encoder_input: batch_x, lstm_input: batch_seq, bn_train: False})

				alpha = 1.0
				beta = 100
				score = (1.0 - d_loss) * 100 + alpha * r_loss + beta * c_loss
				print('outlier Anomaly Score:', score, ', d loss:', d_loss, ', r loss:', r_loss, ', c loss:', c_loss)

				batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, 1)

				d_loss, r_loss, f_loss, c_loss = sess.run([d_fake_output, residual_loss, feature_matching_loss, conceptual_loss],
				feed_dict={d_input: batch_x, g_encoder_input: batch_x,
				lstm_input: batch_seq, bn_train: False})
				score = (1.0 - d_loss) * 10 + alpha * r_loss + beta * c_loss
				print('inlier Anomaly Score:', score, ', d loss:', d_loss, ', r loss:', r_loss, ', c loss:', c_loss)

			try:
				saver.save(sess, './model/SE_ADNet_L.ckpt')
			except:
				print('Save failed')
		else:
			for i in range(100):
				batch_x, batch_seq = util.get_sequence_batch(outlier_sample, lstm_sequence_length, 1)

				# batch_x = np.ones_like(batch_x)

				d_loss, r_loss, f_loss, c_loss = sess.run([d_fake_output, residual_loss, feature_matching_loss, conceptual_loss],
				feed_dict={d_input: batch_x, g_encoder_input: batch_x,
				lstm_input: batch_seq, bn_train: False})
				alpha = 1.0
				beta = 100

				score = (1.0 - d_loss) * 10 + alpha * r_loss + beta * c_loss
				print('outlier Anomaly Score:', score, ', d loss:', d_loss, ', r loss:', r_loss, ', c loss:', c_loss)

				batch_x, batch_seq = util.get_sequence_batch(inlier_sample, lstm_sequence_length, 1)

				d_loss, r_loss, f_loss, c_loss = sess.run([d_fake_output, residual_loss, feature_matching_loss, conceptual_loss],
				feed_dict={d_input: batch_x, g_encoder_input: batch_x,
				lstm_input: batch_seq, bn_train: False})

				score = (1.0 - d_loss) * 10 + alpha * r_loss + beta * c_loss

				print('inlier Anomaly Score:', score, ', d loss:', d_loss , ', r loss:', r_loss, ', c loss:', c_loss)


if __name__ == '__main__':
	#train(pretrain=False, b_test=False)
	train(pretrain=False, b_test=True)
