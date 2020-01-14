import tensorflow as tf
from model import Generator, Discriminator
from utils import get_dataset
import os

def train(FLAGS):

	@tf.function
	def gen_train_one_step():
		z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], 0., 1.)
		fake_labels = tf.one_hot(tf.random.uniform([FLAGS.BATCH_SIZE], maxval = FLAGS.N_CLASS, dtype = tf.int32), depth = FLAGS.N_CLASS, dtype = tf.float32)

		with tf.GradientTape() as tape:
			tape.watch(generator.trainable_variables)
			fake_samples = generator(z, fake_labels)
			fake_logits = discriminator(fake_samples, fake_labels)
			gen_loss = - tf.reduce_mean(fake_logits)
			gradient = tape.gradient(gen_loss, generator.trainable_variables)

		gen_opt.apply_gradients(zip(gradient, generator.trainable_variables))

	@tf.function
	def dis_train_one_step(real_samples, real_labels):
		z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], 0., 1.)
		fake_labels = real_labels

		with tf.GradientTape() as tape:
			tape.watch(discriminator.trainable_variables)

			fake_samples = generator(z, fake_labels)
			real_logits = discriminator(real_samples, real_labels)
			fake_logits = discriminator(fake_samples, fake_labels)

			alpha = tf.random.uniform([FLAGS.BATCH_SIZE, 1, 1, 1])
			inter_samples = fake_samples * alpha + real_samples * (1 - alpha)
			with tf.GradientTape() as gp_tape:
				gp_tape.watch(inter_samples)
				inter_logits = discriminator(inter_samples, real_labels)
				gp_gradient = gp_tape.gradient(inter_logits, inter_samples)
			gp_gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradient), axis = [1, 2, 3]))
			gradient_penalty = tf.reduce_mean((gp_gradient_norm - 1.) ** 2) * FLAGS.GRADIENT_PENALTY_LAMBDA

			gen_loss = - tf.reduce_mean(fake_logits)
			dis_loss = - tf.reduce_mean(real_logits) + tf.reduce_mean(fake_logits) + gradient_penalty
			adv_loss = - tf.reduce_mean(real_logits) + tf.reduce_mean(fake_logits)

			gradient = tape.gradient(dis_loss, discriminator.trainable_variables)

		dis_opt.apply_gradients(zip(gradient, discriminator.trainable_variables))
		gen_metric(gen_loss)
		dis_metric(dis_loss)
		adv_metric(adv_loss)

	def logging(step):
		with writer.as_default():
			tf.summary.scalar('adv_loss', adv_metric.result(), step = step)
			tf.summary.scalar('gen_loss', gen_metric.result(), step = step)
			tf.summary.scalar('dis_loss', dis_metric.result(), step = step)
			adv_metric.reset_states()
			gen_metric.reset_states()
			dis_metric.reset_states()

			z = tf.random.uniform([FLAGS.BATCH_SIZE, FLAGS.NOISE_DIM], 0., 1.)
			fake_labels = tf.one_hot(tf.random.uniform([FLAGS.BATCH_SIZE], maxval = FLAGS.N_CLASS, dtype = tf.int32), depth = FLAGS.N_CLASS)
			fake_samples = generator(z, fake_labels)
			tf.summary.image('fake_samples', (fake_samples + 1) / 2, step = step, max_outputs = 10)

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
				logging(chkpt.step)

			if chkpt.step % FLAGS.CHECKPOINT_INTERVAL == 0:
				manager.save(checkpoint_number = chkpt.step)

	manager.save(checkpoint_number = chkpt.step)