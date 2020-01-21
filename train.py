import tensorflow as tf
from model import Generator, Discriminator # , Generator_v1, Discriminator_v1
from utils import get_dataset
import os

def train(FLAGS):

	# @tf.function
	def gen_train_one_step():
		z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], -1., 1.)
		fake_labels = tf.one_hot(tf.random.uniform([FLAGS.BATCH_SIZE], maxval = FLAGS.N_CLASS, dtype = tf.int32), depth = FLAGS.N_CLASS, dtype = tf.float32)

		with tf.GradientTape() as tape:
			tape.watch(generator.trainable_variables)
			fake_samples = generator(z, fake_labels, train = True)
			fake_logits = discriminator(fake_samples, fake_labels, train = False)
			gen_loss = - tf.reduce_mean(fake_logits)
			gradient = tape.gradient(gen_loss, generator.trainable_variables)

		gen_opt.apply_gradients(zip(gradient, generator.trainable_variables))

	# @tf.function
	def dis_train_one_step(real_samples, real_labels):
		z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], -1., 1.)
		fake_labels = real_labels

		with tf.GradientTape() as tape:
			tape.watch(discriminator.trainable_variables)

			fake_samples = generator(z, fake_labels, train = False)

			real_logits = discriminator(real_samples, real_labels, train = False)
			fake_logits = discriminator(fake_samples, fake_labels, train = True)

			dis_loss = tf.reduce_mean(tf.maximum(1 - real_logits, 0)) + tf.reduce_mean(tf.maximum(1 + fake_logits, 0))
			gen_loss = - tf.reduce_mean(fake_logits)
			adv_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

			gradient = tape.gradient(dis_loss, discriminator.trainable_variables)

		dis_opt.apply_gradients(zip(gradient, discriminator.trainable_variables))
		gen_metric(gen_loss)
		dis_metric(dis_loss)
		adv_metric(adv_loss)

	def logging(step, real_samples):
		with writer.as_default():
			tf.summary.scalar('adv_loss', adv_metric.result(), step = step)
			tf.summary.scalar('gen_loss', gen_metric.result(), step = step)
			tf.summary.scalar('dis_loss', dis_metric.result(), step = step)
			adv_metric.reset_states()
			gen_metric.reset_states()
			dis_metric.reset_states()

			z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], -1., 1.)
			# fake_labels = tf.one_hot(tf.random.uniform([FLAGS.BATCH_SIZE], maxval = FLAGS.N_CLASS, dtype = tf.int32), depth = FLAGS.N_CLASS)
			fake_labels = tf.one_hot(tf.range(FLAGS.BATCH_SIZE, dtype = tf.int32) % 10, depth = 10, dtype = tf.float32)
			fake_samples = generator(z, fake_labels, train = False)
			tf.summary.image('fake_samples', (fake_samples + 1) / 2, step = step, max_outputs = 9)
			tf.summary.image('real_samples', (real_samples + 1) / 2, step = step, max_outputs = 9)

			def diff(x):
				return tf.math.abs(x[0] - x[1])
			# tf.summary.histogram('dis_weight', diff(discriminator.layer_stack[3].linear_gamma.weights[0]), step = step)
			# tf.summary.histogram('gen_weight', diff(generator.layer_stack[2].cBN1.linear_gamma.weights[0]), step = step)

			writer.flush()

	# Initialize Dataset
	dataset = get_dataset(FLAGS)

	# Initialize Model
	generator = Generator()
	discriminator = Discriminator()

	# Initialize Optimizer
	gen_opt = tf.keras.optimizers.Adam(FLAGS.GEN_LEARNING_RATE, FLAGS.ADAM_BETA_1, FLAGS.ADAM_BETA_2)
	dis_opt = tf.keras.optimizers.Adam(FLAGS.DIS_LEARNING_RATE, FLAGS.ADAM_BETA_1, FLAGS.ADAM_BETA_2)

	# Initilize Metrics
	gen_metric = tf.keras.metrics.Mean(name = 'Generator_Loss')
	dis_metric = tf.keras.metrics.Mean(name = 'Discriminator_Loss')
	adv_metric = tf.keras.metrics.Mean(name = 'Adversarial_Loss')

	# Initialize Log Writer
	writer = tf.summary.create_file_writer(os.path.join(FLAGS.LOG_DIR, FLAGS.MODEL_PREFIX + '.log'))

	# Initialize Checkpoint Manager
	chkpt = tf.train.Checkpoint(step = tf.Variable(0, dtype = tf.int64), 
								gen_optimizer = gen_opt,
								dis_optimizer = dis_opt,
								generator = generator,
								discriminator = discriminator)
	manager = tf.train.CheckpointManager(chkpt, FLAGS.CHECKPOINT_DIR, 
										max_to_keep = 3, 
										checkpoint_name = FLAGS.MODEL_PREFIX)

	# Restore Checkpoint
	chkpt.restore(manager.latest_checkpoint)

	for i in range(FLAGS.MAX_EPOCH):

		for (sample, label) in dataset:
			chkpt.step.assign_add(1)
			dis_train_one_step(sample, label)

			if chkpt.step % FLAGS.N_UPDATE_DIS == 0:
				gen_train_one_step()

			if chkpt.step % FLAGS.LOG_INTERVAL == 0:
				logging(chkpt.step, sample)

			if chkpt.step % FLAGS.CHECKPOINT_INTERVAL == 0:
				manager.save(checkpoint_number = chkpt.step)

	manager.save(checkpoint_number = chkpt.step)