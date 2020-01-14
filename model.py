import tensorflow as tf

class ConditionalShift(tf.keras.layers.Layer):
	def __init__(self, channel, **kwargs):
		super(ConditionalShift, self).__init__(**kwargs)
		self.linear_gamma = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.ones,
			name = 'linear_gamma')
		self.linear_beta = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.zeros,
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
			name = 'linear_gamma')
		self.linear_beta = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.zeros,
			name = 'linear_beta')
		self.batchnorm = tf.keras.layers.BatchNormalization()

	def call(self, x, c):
		n_dim = len(x.get_shape().as_list())
		x = self.batchnorm(x)
		gamma = self.linear_gamma(c)
		beta = self.linear_beta(c)
		for i in range(n_dim - 2):
			gamma = tf.expand_dims(gamma, axis = 1)
			beta = tf.expand_dims(beta, axis = 1)
		return x * gamma + beta

class Generator(tf.keras.Model):
	def __init__(self, **kwargs):
		super(Generator, self).__init__(**kwargs)
		self.layer_stack = []
		self.layer_stack.append(tf.keras.layers.Dense(7 * 7 * 64, activation = tf.nn.relu, name = 'fc1'))
		self.layer_stack.append(tf.keras.layers.Reshape([7, 7, 64]))
		self.layer_stack.append(ConditionalBatchNorm(64, name = 'bn1'))
		self.layer_stack.append(tf.keras.layers.Conv2DTranspose(64, 5, 2, 'same', activation = tf.nn.relu, name = 'deconv1'))
		self.layer_stack.append(ConditionalBatchNorm(64, name = 'bn2'))
		self.layer_stack.append(tf.keras.layers.Conv2DTranspose(64, 5, 2, 'same', activation = tf.nn.relu, name = 'deconv2'))
		self.layer_stack.append(ConditionalBatchNorm(64, name = 'bn3'))
		self.layer_stack.append(tf.keras.layers.Conv2DTranspose(1, 5, 1, 'same', activation = tf.nn.tanh, name = 'deconv3'))

	def call(self, x, c):
		for layer in self.layer_stack:
			if isinstance(layer, ConditionalBatchNorm):
				x = layer(x, c)
			else:
				x = layer(x)
		return x

class Discriminator(tf.keras.Model):
	def __init__(self, **kwargs):
		super(Discriminator, self).__init__(**kwargs)
		self.layer_stack = []
		self.layer_stack.append(tf.keras.layers.Conv2D(64, 5, 1, 'same', activation = tf.nn.leaky_relu, name = 'conv1'))
		self.layer_stack.append(ConditionalShift(64, name = 'cs1'))
		self.layer_stack.append(tf.keras.layers.Conv2D(64, 5, 2, 'same', activation = tf.nn.leaky_relu, name = 'conv2'))
		self.layer_stack.append(ConditionalShift(64, name = 'cs2'))
		self.layer_stack.append(tf.keras.layers.Conv2D(64, 5, 2, 'same', activation = tf.nn.leaky_relu, name = 'conv3'))
		self.layer_stack.append(ConditionalShift(64, name = 'cs3'))
		self.layer_stack.append(tf.keras.layers.Flatten())
		self.layer_stack.append(tf.keras.layers.Dense(32, activation = tf.nn.leaky_relu, name = 'fc1'))
		self.layer_stack.append(tf.keras.layers.Dense(1, name = 'fc2'))

	def call(self, x, c):
		for layer in self.layer_stack:
			if isinstance(layer, ConditionalShift):
				x = layer(x, c)
			else:
				x = layer(x)
		return x