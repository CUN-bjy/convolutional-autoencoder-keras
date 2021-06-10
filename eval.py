import tensorflow as tf
from autoencoder import AutoEncoder


import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
	# print(x_train.shape, y_train.shape)

	x_train = x_train.reshape(-1, 28, 28, 1)
	x_train = x_train / 127.5 - 1 # Normalization
	# print(x_train.min(), x_train.max())

	checkpoint_path = 'tmp/01-basic-auto-encoder-MNIST.ckpt'

	ae = AutoEncoder(in_dim=(28,28,1),latent_dim=(5,))
	# ae.encoder.summary();	ae.decoder.summary();	ae.auto_encoder.summary()

	while True:
		# load latest model
		ae.auto_encoder.load_weights(checkpoint_path)

		# original figure
		fig, axes = plt.subplots(3, 5)
		fig.set_size_inches(12, 6)
		for i in range(15):
			axes[i//5, i%5].imshow(x_train[i].reshape(28, 28), cmap='gray')
			axes[i//5, i%5].axis('off')
		plt.tight_layout()
		plt.savefig("original.png")

		# evaluation figure
		fig, axes = plt.subplots(3, 5)
		fig.set_size_inches(12, 6)
		for i in range(15):
			img = ae.auto_encoder.predict(x_train[i].reshape(-1,28,28,1))
			axes[i//5, i%5].imshow(img.reshape(28,28), cmap='gray')
			axes[i//5, i%5].axis('off')
		plt.tight_layout()
		plt.savefig("decoded.png")
		print("Evaluation Done.")
		time.sleep(20)

		# latent space visualization
		# xy = ae.encoder.predict(x_train)

		# plt.figure(figsize=(15, 12))
		# plt.scatter(x=xy[:, 0], y=xy[:, 1], c=y_train, cmap=plt.get_cmap('Paired'), s=3)
		# plt.colorbar()
		# plt.savefig("mygraph.png")