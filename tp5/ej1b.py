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
import imageio
from PIL import Image

INPUT_SIZE = 35
LATENT_SIZE = 2
HIDDEN_SIZE = 15

# Using ReLU here causes overflow in next hidden layer Sigmoid function
LATENT_FUNCTION = Sigmoid()

fonts_headers = np.array(
    ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
     "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]
)


def graph_fonts(original, decoded, char_name=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(f'Original ({char_name})')
    ax2.set_title(f'Decoded')
    ax1.imshow(np.array(original).reshape((7, 5)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((7, 5)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
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
    def __init__(self, amount_correct_characters, decoder_outputs, autoencoder: Autoencoder,
                 recognized_characters: list):
        self.amount_correct_characters = amount_correct_characters
        self.decoder_outputs = decoder_outputs
        self.autoencoder = autoencoder
        self.recognized_characters = recognized_characters


def calculate_results(autoencoder: Autoencoder, dataset_input: dict) -> Results:
    dataset_input_list = list(dataset_input.values())

    decoder_outputs = []
    amount_correct_characters = 0
    recognized_characters = []

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
            recognized_characters.append(fonts_headers[i])

    return Results(amount_correct_characters, decoder_outputs, autoencoder, recognized_characters)


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


def print_noice_example(noice_prob: float):
    dataset = list(load_pattern_map('characters.txt').values())
    noisy_dataset = [salt_and_pepper_noise(pattern, salt_prob=noice_prob, pepper_prob=noice_prob) for pattern in
                     dataset]
    for i in range(len(dataset)):
        graph_fonts(dataset[i], noisy_dataset[i])


RUNS_MAGIC_NUMBER = 30


def run_different_noise_probabilities():
    results = []

    noise_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Valores variables de noise_prob
    eta = 1e-2
    # uncomment to see the noise example
    # print_noice_example(noise_prob)

    for noise_prob in noise_probs:
        print(f"Noise Probability: {noise_prob}")
        run_results = []
        for i in range(RUNS_MAGIC_NUMBER):
            print(f"Run {i}")
            run_results.append(run(epochs=3000, eta=eta, salt_prob=noise_prob, pepper_prob=noise_prob))
            print()
        results.append(run_results)

    correct_characters_averages = [np.average([result.amount_correct_characters for result in run_results]) for
                                   run_results in results]
    correct_characters_stds = [np.std([result.amount_correct_characters for result in run_results]) for run_results in
                               results]

    # Plotting the results
    plt.errorbar(noise_probs, correct_characters_averages, yerr=correct_characters_stds, fmt='-o')
    plt.xlabel('Noise Probability')
    plt.ylabel('Average Amount of Correct Characters')
    plt.title('Average Amount of Correct Characters with Standard Deviation')
    plt.grid(True)
    plt.show()


def run_different_etas():
    results = []

    etas = [1e-1, 1e-2, 1e-3]  # Valores variables de eta
    noise_prob = 0.1  # Fija un valor de noise_prob

    for eta in etas:
        print(f"Eta: {eta}")
        run_results = []
        for i in range(RUNS_MAGIC_NUMBER):
            print(f"Run {i}")
            run_results.append(run(epochs=3000, eta=eta, salt_prob=noise_prob, pepper_prob=noise_prob))
            print()
        results.append(run_results)

    correct_characters_averages = [np.average([result.amount_correct_characters for result in run_results]) for
                                   run_results in results]
    correct_characters_stds = [np.std([result.amount_correct_characters for result in run_results]) for run_results in
                               results]

    # Plotting the results
    x_coords = np.arange(len(etas))
    plt.bar(x_coords, correct_characters_averages, yerr=correct_characters_stds, align='center', alpha=0.5,
            ecolor='black', capsize=10)
    plt.xlabel('Eta')
    plt.xticks(x_coords, etas)
    plt.ylabel('Average Amount of Correct Characters')
    plt.title('Average Amount of Correct Characters with Standard Deviation')
    plt.grid(True)
    plt.show()


def run_noise_character_detection():
    noise_prob = 0.1
    eta = 1e-2
    # uncomment to see the noise example
    # print_noice_example(noise_prob)

    result = run(epochs=9000, eta=eta, salt_prob=noise_prob, pepper_prob=noise_prob)

    print(f"Recognized characters: {result.recognized_characters}")

    dataset = load_pattern_map('characters.txt')
    recognized_characters = []
    for key, value in dataset.items():
        if key in result.recognized_characters:
            recognized_characters.append(value)
    for i in range(len(recognized_characters)):
        noisy_char = salt_and_pepper_noise(recognized_characters[i], salt_prob=noise_prob, pepper_prob=noise_prob)
        char = result.autoencoder.feedforward(noisy_char)
        for j in range(len(char)):
            if char[j][0] >= 0:
                char[j][0] = 1
            else:
                char[j][0] = -1
        graph_fonts(noisy_char, char, result.recognized_characters[i])


def run_noise_over_character_n_times(n: int, char: str):
    dataset = load_pattern_map('characters.txt')
    noise_prob = 0.1

    c_char = dataset[char]

    # List to store each frame for gif
    c_frames = []

    for i in range(n):
        noisy_c_char = salt_and_pepper_noise(c_char, salt_prob=noise_prob, pepper_prob=noise_prob)

        # Convert matrices to images and append to frames list
        # Normalize values from -1-1 to 0-255
        c_frame = Image.fromarray(((noisy_c_char.reshape(7, 5) + 1) * 0.5 * 255).astype(np.uint8))

        # Resize images, e.g., to 140x100 while keeping aspect ratio
        c_frame = c_frame.resize((560, 400), Image.NEAREST)

        c_frames.append(c_frame)

    # Save frames as gif
    imageio.mimsave('noisy_c_char.gif', c_frames, duration=0.001, loop=0)


if __name__ == '__main__':
    # run_different_noise_probabilities()
    # run_noise_character_detection()
    # run_different_etas()
    run_noise_over_character_n_times(999, 'f')
