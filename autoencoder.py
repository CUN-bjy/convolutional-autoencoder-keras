from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

class AutoEncoder():
	def __init__(self, in_dim, latent_dim):
		self.encoder = self._encoder(in_dim, latent_dim)
		self.decoder = self._decoder(in_dim, latent_dim)

		encoder_in = Input(shape=in_dim)
		x = self.encoder(encoder_in)
		decoder_out = self.decoder(x)
		self.auto_encoder = Model(encoder_in, decoder_out, name="auto_encoder")



	def train(self, x_train, batch_size_, lr_, epoch_):
		self.auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(lr_), loss=tf.keras.losses.MeanSquaredError())

		# checkpoint setting
		checkpoint_path = 'tmp/01-basic-auto-encoder-MNIST.ckpt'
		checkpoint = ModelCheckpoint(checkpoint_path, 
		                             save_best_only=True, 
		                             save_weights_only=True, 
		                             monitor='loss', 
		                             verbose=1)

		# train the autoencoder
		self.auto_encoder.fit(x_train, x_train, 
	                 batch_size=batch_size_, 
	                 epochs=epoch_, 
	                 callbacks=[checkpoint], 
	                )

	def _encoder(self, in_dim, latent_dim):
		encoder_input = Input(shape=in_dim)

		# W X H -> W X H
		x = Conv2D(32, 3, padding='same')(encoder_input) 
		x = BatchNormalization()(x)
		x = LeakyReLU()(x) 

		# W X H -> W/2 X H/2
		x = Conv2D(64, 3, strides=2, padding='same')(x)
		x = BatchNormalization()(x) 
		x = LeakyReLU()(x) 

		# W/2 X H/2 -> W/4 X H/4
		x = Conv2D(64, 3, strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		# W/4 X H/4 -> W/4 X H/4
		x = Conv2D(64, 3, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		# encode to latent space (Z)
		encoder_output = Dense(latent_dim[0])(x)
		return Model(encoder_input, encoder_output, name='encoder')

	def _decoder(self, in_dim, latent_dim):
		decoder_input = Input(shape=latent_dim)

		# latent space to state space
		x = Dense(int(in_dim[0]/4)*int(in_dim[1]/4)*64)(decoder_input)
		x = Reshape( (int(in_dim[0]/4), int(in_dim[1]/4), 64))(x)

		# W/4 X H/4 -> W/4 X H/4
		x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		# W/4 X H/4 -> W/2 X H/2
		x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		# W/2 X H/2 -> W X H
		x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		# W X H -> W X H
		x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		# 최종 output
		decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)
		return Model(decoder_input, decoder_output, name='decoder')