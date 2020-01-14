import tensorflow as tf

def get_dataset(FLAGS):
	(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5)
	train_labels = tf.one_hot(train_labels, depth = FLAGS.N_CLASS, dtype = tf.float32)
	train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(FLAGS.BATCH_SIZE, drop_remainder = True)
	return train_dataset

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	FLAGS = parser.parse_args()
	FLAGS.BATCH_SIZE = 64
	get_dataset(FLAGS)