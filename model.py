import tensorflow as tf
from functools import partial

class SN(tf.keras.layers.Wrapper):
	def __init__(self, layer, **kwargs):
		super(SN, self).__init__(layer, **kwargs)
		self.layer = layer

	def build(self, input_shape):
		if not self.layer.built:
			self.layer.build(input_shape)

			self.w = self.layer.kernel
			self.w_shape = self.w.get_shape().as_list()
			self.u = self.add_weight("u",
									shape = [self.w_shape[-1], 1],
									initializer = tf.random.normal,
									trainable = False)
		super(SN, self).build()

	def call(self, x, train = True):

		def _power_iteration(w, train):
			_v = tf.matmul(w, self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(tf.transpose(w), v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if train:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(v), w), u))
			return w

		w = tf.reshape(self.w, [-1, self.w_shape[-1]])
		self.layer.kernel = tf.reshape(_power_iteration(w, train = train), self.w_shape)
		return self.layer(x)

class ConditionalShift(tf.keras.layers.Layer):
	def __init__(self, channel, **kwargs):
		super(ConditionalShift, self).__init__(**kwargs)
		self.linear_gamma = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.ones, # partial(tf.random.normal, mean = 1.0, stddev = 0.01),
			use_bias = False,
			name = 'linear_gamma')
		self.linear_beta = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.zeros, # partial(tf.random.normal, mean = 0.0, stddev = 0.01),
			use_bias = False,
			name = 'linear_beta')
	def call(self, x, c):
		n_dim = len(x.get_shape().as_list())
		gamma = self.linear_gamma(c)
		beta = self.linear_beta(c)
		for i in range(n_dim - 2):
			gamma = tf.expand_dims(gamma, axis = 1)
			beta = tf.expand_dims(beta, axis = 1)
		return x * gamma + beta

class ConditionalBatchNorm(tf.keras.layers.Layer):
	def __init__(self, channel, **kwargs):
		super(ConditionalBatchNorm, self).__init__(**kwargs)
		self.linear_gamma = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.ones,
			use_bias = False,
			name = 'linear_gamma')
		self.linear_beta = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.zeros,
			use_bias = False,
			name = 'linear_beta')
		self.batchnorm = tf.keras.layers.BatchNormalization()

	def call(self, x, c, train = True):
		n_dim = len(x.get_shape().as_list())
		x = self.batchnorm(x)
		gamma = self.linear_gamma(c)
		beta = self.linear_beta(c)
		for i in range(n_dim - 2):
			gamma = tf.expand_dims(gamma, axis = 1)
			beta = tf.expand_dims(beta, axis = 1)
		return x * gamma + beta

class GResBlock(tf.keras.layers.Layer):
	def __init__(self, channel, upsample = True, **kwargs):
		super(GResBlock, self).__init__(**kwargs)
		self.deconv1 = SN(tf.keras.layers.Conv2DTranspose(channel, 5, 2 if upsample else 1, 'same', name = 'deconv1', activation = tf.nn.leaky_relu))
		self.cBN1 = ConditionalShift(channel)
		self.deconv2 = SN(tf.keras.layers.Conv2DTranspose(channel, 5, 1, 'same', name = 'deconv2', activation = tf.nn.leaky_relu))
		self.cBN2 = ConditionalShift(channel)
		self.upsample = tf.keras.layers.UpSampling2D(2 if upsample else 1)
		self.deconv3 = SN(tf.keras.layers.Conv2DTranspose(channel, 1, 1, 'same', name = 'deconv3', activation = tf.nn.leaky_relu))
	def call(self, x, c, train = True):
		y = self.deconv1(x, train = train)
		y = self.cBN1(y, c)
		y = self.deconv2(y, train = train)
		y = self.cBN2(y, c)
		x = self.deconv3(self.upsample(x), train = train)
		return x + y

class Generator(tf.keras.Model):
	def __init__(self, **kwargs):
		super(Generator, self).__init__(**kwargs)
		self.layer_stack = []
		self.layer_stack.append(SN(tf.keras.layers.Dense(7 * 7 * 256, activation = tf.nn.leaky_relu, name = 'fc1')))
		self.layer_stack.append(tf.keras.layers.Reshape([7, 7, 256]))
		self.layer_stack.append(GResBlock(128))
		self.layer_stack.append(GResBlock(128))
		self.layer_stack.append(SN(tf.keras.layers.Conv2DTranspose(1, 5, 1, 'same', activation = tf.nn.tanh, name = 'out1')))

	def call(self, x, c, train = True):
		for layer in self.layer_stack:
			if isinstance(layer, SN):
				x = layer(x, train = train)
			elif isinstance(layer, GResBlock):
				x = layer(x, c, train = train)
			else:
				x = layer(x)
		return x

class Discriminator(tf.keras.Model):
	def __init__(self, **kwargs):
		super(Discriminator, self).__init__(**kwargs)
		self.layer_stack = []
		self.layer_stack.append(SN(tf.keras.layers.Conv2D(128, 5, 1, 'same', activation = tf.nn.leaky_relu, name = 'conv1')))
		# self.layer_stack.append(ConditionalShift(64, name = 'cs1'))
		self.layer_stack.append(SN(tf.keras.layers.Conv2D(128, 5, 2, 'same', activation = tf.nn.leaky_relu, name = 'conv2')))
		# self.layer_stack.append(ConditionalShift(64, name = 'cs2'))
		self.layer_stack.append(SN(tf.keras.layers.Conv2D(128, 5, 2, 'same', activation = tf.nn.leaky_relu, name = 'conv3')))
		
		# self.layer_stack.append(global_sum_pooling)
		self.layer_stack.append(tf.keras.layers.Flatten())
		self.layer_stack.append(SN(tf.keras.layers.Dense(32, activation = tf.nn.leaky_relu, name = 'fc1')))
		self.layer_stack.append(SN(tf.keras.layers.Dense(1, name = 'fc2')))

	def call(self, x, c, train = True):
		c = tf.tile(tf.reshape(c, [-1, 1, 1, 10]), [1, 28, 28, 1])
		x = tf.concat([x, c], axis = 3)
		for layer in self.layer_stack:
			if isinstance(layer, SN):
				x = layer(x, train = train)
			else:
				x = layer(x)
		return x
