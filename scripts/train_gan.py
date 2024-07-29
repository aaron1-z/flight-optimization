from utils.data_processing import load_and_preprocess_data
from models.gan import build_generator, build_discriminator, train_gan
import tensorflow as tf

data, data_scaled, scaler = load_and_preprocess_data('data/flight_data.csv')

input_dim = data_scaled.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(input_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

train_gan(generator, discriminator, gan, data_scaled)
