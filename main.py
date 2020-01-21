import argparse
from train import train
import subprocess
import os

parser = argparse.ArgumentParser(description = 'GAN Training', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Directories
group_parser = parser.add_argument_group('Directory Configuration')
group_parser.add_argument("--LOG_DIR", type = str, default = './logs/', help = "Logging Directory")
group_parser.add_argument('--LOG_INTERVAL', type = int, default = 100, help = 'Number of Steps for One Log')
group_parser.add_argument('--CHECKPOINT_INTERVAL', type = int, default = 5000, help = 'Number of Steps for One Checkpoint')
group_parser.add_argument("--CHECKPOINT_DIR", type = str, default = './chkpts/', help = "Checkpoint Directory")
group_parser.add_argument("--MODEL_PREFIX", type = str, default = 'default', help = "Checkpoint Model Prefix")

# Hardwave Configuration
group_parser = parser.add_argument_group('Hardware Configuration')
group_parser.add_argument("--GPU_MODE", action = 'store_true', help = "Whether Use GPU")
group_parser.add_argument('--VISIBLE_DEVICE', type = str, default = None, help = 'Visible Devices')

# Model Configuration
group_parser = parser.add_argument_group('Model Configuration')
group_parser.add_argument('--NOISE_DIM', type = int, default = 64, help = 'Dimension of Noise')
group_parser.add_argument('--N_CLASS', type = int, default = 10, help = 'Number of Classes')

# Training Hyperparameters
group_parser = parser.add_argument_group('Training Configuration')
group_parser.add_argument('--MAX_EPOCH', type = int, default = 400, help = 'Maximal Number of Epoches')
group_parser.add_argument('--BATCH_SIZE', type = int, default = 64, help = 'Batch Size')
group_parser.add_argument('--GEN_LEARNING_RATE', type = float, default = 5e-4, help = 'Learning Rate of Generator')
group_parser.add_argument('--DIS_LEARNING_RATE', type = float, default = 5e-4, help = 'Learning Rate of Discriminator')
group_parser.add_argument('--N_UPDATE_DIS', type = int, default = 1, help = 'Number of Updates of Dis per one for GEN and CLS')
group_parser.add_argument('--GRADIENT_PENALTY_LAMBDA', type = int, default = 10, help = 'Weight of Gradient Penalty')
group_parser.add_argument('--ADAM_BETA_1', type = float, default = 0.5, help = 'Parameter Beta_1 of Adam Optimizer')
group_parser.add_argument('--ADAM_BETA_2', type = float, default = 0.999, help = 'Parameter Beta_2 of Adam Optimizer')

def configure_gpu(FLAGS):
	# GPU Mode
	gpus = []
	for i in range(16):
		try:
			out = subprocess.check_output(['nvidia-smi', '-i', str(i)]).decode('utf-8').split('\n')
			if out[15].find('No running') != -1:
				gpus.append(str(i))
				break
		except:
			break

	if len(gpus) == 0:
		print ('[ERROR] Problems Occurred at GPU Configuration')
		exit(1)
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
	FLAGS.VISIBLE_DEVICE = ','.join(gpus)

def hardware_configure(FLAGS):
	import os
	if not FLAGS.GPU_MODE:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
	else:
		configure_gpu(FLAGS)
		
def print_config(FLAGS):
	default_flags = parser.parse_args([])
	with open(FLAGS.MODEL_PREFIX + '-config.log', 'w') as f:
		for index in FLAGS.__dict__:
			if FLAGS.__dict__[index] == default_flags.__dict__[index]:
				print ('{}: {}'.format(index, FLAGS.__dict__[index]), file = f)
			else:
				print ('{}: {} (Changed)'.format(index, FLAGS.__dict__[index]), file = f)

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	hardware_configure(FLAGS)
	print_config(FLAGS)
	train(FLAGS)