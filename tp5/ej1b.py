from tp5.optimizer import Adam, Optimizer
from tp5.nn import MLP, NoiseFunctionType
from tp5.layers import Dense
from tp5.activations import ReLU, Sigmoid, Tanh
from tp5.autoencoder import Autoencoder
from tp5.loss import MSE
from tp4.Hopfield.pattern_loader import *
import matplotlib.pyplot as plt
import numpy as np
from time import time

INPUT_SIZE = 35
LATENT_SIZE = 2
HIDDEN_SIZE = 15

# Using ReLU here causes overflow in next hidden layer Sigmoid function
LATENT_FUNCTION = Sigmoid()

fonts_headers = np.array(
    ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
     "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]
)


def graph_fonts(original, decoded):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')

    ax1.imshow(np.array(original).reshape((7, 5)), cmap='gray')
    ax1.set_xticks(np.arange(0, 5, 1))
    ax1.set_yticks(np.arange(0, 7, 1))
    ax1.set_xticklabels(np.arange(1, 6, 1))
    ax1.set_yticklabels(np.arange(1, 8, 1))

    ax2.imshow(np.array(decoded).reshape((7, 5)), cmap='gray')
    ax2.set_xticks(np.arange(0, 5, 1))
    ax2.set_yticks(np.arange(0, 7, 1))
    ax2.set_xticklabels(np.arange(1, 6, 1))
    ax2.set_yticklabels(np.arange(1, 8, 1))

    fig.show()


def salt_and_pepper_noise(input: np.ndarray, salt_prob=0.1, pepper_prob=0.1) -> np.ndarray:
    noisy_image = input.copy()
    total_pixels = input.size

    # Number of 'salt' and 'pepper' pixels to add
    num_salt = np.ceil(total_pixels * salt_prob)
    num_pepper = np.ceil(total_pixels * pepper_prob)

    # Add 'salt' noise
    salt_coords = [np.random.randint(0, i, int(num_salt)) for i in input.shape]
    noisy_image[tuple(salt_coords)] = 1

    # Add 'pepper' noise
    pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in input.shape]
    noisy_image[tuple(pepper_coords)] = -1

    return noisy_image


def create_autoencoder(noise: NoiseFunctionType, optimizer: Optimizer = Adam(1e-2)) -> Autoencoder:
    encoder = MLP()
    # 35 -> 15 -> 2
    encoder.addLayer(Dense(inputDim=INPUT_SIZE, outputDim=HIDDEN_SIZE, activation=Sigmoid(), optimizer=optimizer))
    encoder.addLayer(
        Dense(inputDim=HIDDEN_SIZE, outputDim=LATENT_SIZE, activation=LATENT_FUNCTION, optimizer=optimizer))

    decoder = MLP()
    # 2 -> 15 -> 35
    decoder.addLayer(Dense(inputDim=LATENT_SIZE, outputDim=HIDDEN_SIZE, activation=Sigmoid(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE, outputDim=INPUT_SIZE, activation=Tanh(), optimizer=optimizer))

    autoencoder = Autoencoder(encoder, decoder, noise=noise)

    return autoencoder


class Results:
    def __init__(self, amount_correct_characters, decoder_outputs, dots):
        self.amount_correct_characters = amount_correct_characters
        self.decoder_outputs = decoder_outputs
        self.dots = dots


def calculate_results(autoencoder: Autoencoder, dataset_input: dict) -> Results:
    dataset_input_list = list(dataset_input.values())

    dots = []
    decoder_outputs = []
    amount_correct_characters = 0

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output_history = []
        output = autoencoder.feedforward(input_reshaped, output_history)
        for j in range(len(output)):
            if output[j][0] >= 0:
                output[j][0] = 1
            else:
                output[j][0] = -1

        decoder_outputs.append(output)

        different_pixels = np.where(output.flatten() != dataset_input_list[i])
        amount_different_pixels = len(different_pixels[0])

        if amount_different_pixels <= 1:
            amount_correct_characters += 1

        # First index: 1 because latent space is in index 1
        # Second index: 0 and 1 represent x and y respectively
        # Third index: 0 because is a list of list
        dot = (output_history[1][0][0], output_history[1][1][0])
        dots.append(dot)

    return Results(amount_correct_characters, decoder_outputs, dots)


def run(eta=1e-2, epochs=5000, salt_prob=0.1, pepper_prob=0.1):
    start_time = time()
    dataset_input = load_pattern_map('characters.txt')
    noise = lambda input: salt_and_pepper_noise(input, salt_prob=salt_prob, pepper_prob=pepper_prob)

    autoencoder = create_autoencoder(optimizer=Adam(eta), noise=noise)

    my_callbacks = {}  # {"loss": loss_callback}

    autoencoder.train(dataset_input=list(dataset_input.values()), loss=MSE(), metrics=["train_loss", "test_loss"],
                      tensorboard=False, epochs=epochs, callbacks=my_callbacks)

    results = calculate_results(autoencoder, dataset_input)
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Amount of correct characters: {results.amount_correct_characters}")
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"Epochs: {epochs}")
    return results


RUNS_MAGIC_NUMBER = 30

if __name__ == '__main__':
    results = []

    for i in range(RUNS_MAGIC_NUMBER):
        print(f"Run {i}")
        results.append(run(epochs=5000))
        print()

    correct_characters_average = np.average([result.amount_correct_characters for result in results])
    correct_characters_std = np.std([result.amount_correct_characters for result in results])

    print(f"Average amount of correct characters: {correct_characters_average}")
    print(f"Standard deviation of amount of correct characters: {correct_characters_std}")
