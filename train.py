#
# Author : wnsdlqjtm@naver.com
#

import tensorflow as tf
from autoencoder import AutoEncoder


# Hyperparameter
LEARNING_RATE = 0.001
BATCH_SIZE = 128
MAX_EPOCH = 30

if __name__ == '__main__':
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
	# print(x_train.shape, y_train.shape)

	x_train = x_train.reshape(-1, 28, 28, 1)
	x_train = x_train / 127.5 - 1 # Normalization
	# print(x_train.min(), x_train.max())

	ae = AutoEncoder(in_dim=(28,28,1),latent_dim=(5,))
	ae.encoder.summary();	ae.decoder.summary();	ae.auto_encoder.summary()

	ae.train(x_train, BATCH_SIZE, LEARNING_RATE, MAX_EPOCH)