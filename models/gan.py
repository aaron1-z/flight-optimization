import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_generator(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(256, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    return model

def build_discriminator(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, data.shape[1]))
        generated_data = generator.predict(noise)

        real_data = data[np.random.randint(0, data.shape[0], batch_size)]
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, data.shape[1]))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}] [G loss: {g_loss}]")
