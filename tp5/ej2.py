import numpy as np
from tp5.optimizer import Adam
from tp5.nn import MLP
from tp5.layers import Dense
from tp5.activations import ReLU, Sigmoid, Tanh
from tp5.autoencoder import Autoencoder
from tp5.loss import MSE
from tp5.vae import *
from tp4.Hopfield.pattern_loader import *
import matplotlib.pyplot as plt

INPUT_ROWS = 7
INPUT_COLS = 5
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
LATENT_SIZE = 2
HIDDEN_SIZE = 15

NOISE = None

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


def graph_latent_space(dots):
    for j in range(dots.__len__()):
        plt.scatter(dots[j][0], dots[j][1])
        plt.annotate(fonts_headers[j], xy=dots[j], xytext=(dots[j][0] + 0.01, dots[j][1] + 0.01), fontsize=12)
    plt.show()


if __name__ == "__main__":
    dataset_input = load_pattern_map('characters.txt')

    # set the learning rate and optimizer for training
    optimizer = Adam(1e-2)

    encoder = MLP()
    encoder.addLayer(Dense(inputDim=INPUT_SIZE, outputDim=HIDDEN_SIZE, activation=Sigmoid(), optimizer=optimizer))
    sampler = Sampler(HIDDEN_SIZE, LATENT_SIZE, optimizer=optimizer)

    decoder = MLP()
    decoder.addLayer(Dense(inputDim=LATENT_SIZE, outputDim=HIDDEN_SIZE, activation=Sigmoid(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE, outputDim=INPUT_SIZE, activation=Tanh(), optimizer=optimizer))

    vae = VAE(encoder, sampler, decoder)

    print(vae)

    my_callbacks = {}  # {"loss": loss_callback}

    vae.train(dataset_input=list(dataset_input.values()), loss=MSE(), metrics=["train_loss", "test_loss"],
              tensorboard=False, epochs=5000,
              callbacks=my_callbacks)

    dataset_input_list = list(dataset_input.values())

    dots = []
    decoder_outputs = []
    amount_correct_characters = 0

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output_history = []
        output = vae.feedforward(input_reshaped, output_history)
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

        print(f"Character: {fonts_headers[i]}")
        print(f"Error in pixels: {amount_different_pixels}")

        # First index: 1 because latent space is in index 1
        # Second index: 0 and 1 represent x and y respectively
        # Third index: 0 because is a list of list
        dot = (output_history[1][0][0], output_history[1][1][0])
        dots.append(dot)

    # Graph of the neural network
    vae.plotGraph()

    # Plot of dataset images
    print(f"Recognized characters: {amount_correct_characters}/{len(dataset_input_list)}")
    # Top 20 because SciView has limit of 29 graphs
    for j in range(20):
        graph_fonts(list(dataset_input.values())[j], decoder_outputs[j])

    # Plot of latent space
    graph_latent_space(dots)

    # ----------------------
    # Generating new samples TODO: New dataset maybe with larger size and testing sampling
    # ----------------------

    n = 10
    digit_size = INPUT_ROWS
    images = np.zeros((INPUT_ROWS, INPUT_COLS*n))

    random_index1 = np.random.randint(0, fonts_headers.size)
    input_reshaped1 = np.reshape(dataset_input_list[random_index1], (len(dataset_input_list[random_index1]), 1))
    vae.feedforward(input_reshaped1)
    img1 = vae.sampler.sample

    random_index2 = np.random.randint(0, fonts_headers.size)
    while random_index1 == random_index2:
        random_index2 = np.random.randint(0, fonts_headers.size)
    input_reshaped2 = np.reshape(dataset_input_list[random_index2], (len(dataset_input_list[random_index2]), 1))
    vae.feedforward(input_reshaped2)
    img2 = vae.sampler.sample

    for i in range(n):
        z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
        output = vae.decoder.feedforward(z)
        for j in range(len(output)):
            if output[j][0] >= 0:
                output[j][0] = 1
            else:
                output[j][0] = -1
        output = output.reshape(INPUT_ROWS, INPUT_COLS)
        images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

    plt.figure(figsize=(10, 10))
    plt.title(f"From {fonts_headers[random_index1]} to {fonts_headers[random_index2]}")
    plt.imshow(images, cmap='gray')
    plt.show()
