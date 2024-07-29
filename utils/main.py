from utils.data_processing import load_and_preprocess_data
from models.gan import build_generator, build_discriminator, train_gan
from models.rl import train_rl_model # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from envs.flight_env import FlightEnv

if __name__ == "__main__":
    data, data_scaled, scaler = load_and_preprocess_data('data/flight_data.csv')

    # GAN Model Training
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

    # RL Model Training
    model = train_rl_model(data_scaled)

    # Evaluate GAN Model
    noise = np.random.normal(0, 1, (100, data_scaled.shape[1]))
    generated_data = generator.predict(noise)

    plt.figure(figsize=(10, 6))
    plt.plot(generated_data[0], label='Generated Path')
    plt.plot(data_scaled[0], label='Real Path')
    plt.legend()
    plt.show()

    # Evaluate RL Model
    env = FlightEnv(data_scaled)
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            break

    print("Final State:", obs)
