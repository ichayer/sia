from tp5.activations import *
from tp5.vae import *
from tp4.Hopfield.pattern_loader import *
from tp5.emojis import emoji_size, emoji_images, emoji_chars, emoji_names
import matplotlib.pyplot as plt

INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
LATENT_SIZE = 20
HIDDEN_SIZE = 100
HIDDEN_SIZE2 = 200
HIDDEN_SIZE3 = 300

EMOJIS_CHOSEN = len(emoji_images)

NOISE = None


def graph_fonts(original, decoded):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')
    ax1.imshow(np.array(original).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax2.imshow(np.array(decoded).reshape((INPUT_ROWS, INPUT_COLS)), cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.show()


if __name__ == "__main__":
    emoji_indexes = np.array([0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 23, 24, 26,
                              28, 29, 31, 32, 33, 35, 36, 38, 39, 41, 46, 48, 50, 51, 54, 55,
                              57, 58, 59, 61, 62, 63, 64, 65, 67, 73, 75, 78, 81, 83, 84, 85,
                              90, 91, 92, 93, 95])

    data = np.array(emoji_images)
    dataset_input = data[emoji_indexes]
    dataset_input_list = list(dataset_input)

    # Set the learning rate and optimizer for training
    optimizer = Adam(0.0001)

    encoder = MLP()
    encoder.addLayer(Dense(inputDim=INPUT_SIZE, outputDim=HIDDEN_SIZE3, activation=ReLU(), optimizer=optimizer))
    encoder.addLayer(Dense(inputDim=HIDDEN_SIZE3, outputDim=HIDDEN_SIZE2, activation=ReLU(), optimizer=optimizer))
    encoder.addLayer(Dense(inputDim=HIDDEN_SIZE2, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))
    sampler = Sampler(HIDDEN_SIZE, LATENT_SIZE, optimizer=optimizer)

    decoder = MLP()
    decoder.addLayer(Dense(inputDim=LATENT_SIZE, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE, outputDim=HIDDEN_SIZE2, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE2, outputDim=HIDDEN_SIZE3, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE3, outputDim=INPUT_SIZE, activation=Sigmoid(), optimizer=optimizer))

    vae = VAE(encoder, sampler, decoder)

    print(vae)

    vae.train(dataset_input=dataset_input_list, loss=MSE(), metrics=["train_loss", "test_loss"],
              tensorboard=False, epochs=1000,
              callbacks={})

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output = vae.feedforward(input_reshaped)

        # Plot of dataset images
        # Top 20 because SciView has limit of 29 graphs
        if i < 15:
            graph_fonts(list(dataset_input)[i], output)

    # ----------------------
    # Generating new samples
    # ----------------------
    for _ in range(15):

        n = 10
        digit_size = INPUT_ROWS
        images = np.zeros((INPUT_ROWS, INPUT_COLS * n))

        random_index1 = np.random.choice(emoji_indexes)
        input_reshaped1 = np.reshape(emoji_images[random_index1], (len(emoji_images[random_index1]), 1))
        vae.feedforward(input_reshaped1)
        img1 = vae.sampler.sample

        random_index2 = np.random.choice(emoji_indexes)
        while random_index1 == random_index2:
            random_index2 = np.random.choice(emoji_indexes)
        input_reshaped2 = np.reshape(emoji_images[random_index2], (len(emoji_images[random_index2]), 1))
        vae.feedforward(input_reshaped2)
        img2 = vae.sampler.sample

        for i in range(n):
            z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
            output = vae.decoder.feedforward(z)
            output = output.reshape(INPUT_ROWS, INPUT_COLS)
            images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

        plt.figure(figsize=(10, 10))
        plt.title(f"From \"{emoji_names[random_index1]}\" "
                  f"to \"{emoji_names[random_index2]}\"")
        plt.imshow(images, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
