from tp5.activations import ReLU, Sigmoid, Tanh
from tp5.vae import *
from tp4.Hopfield.pattern_loader import *
from tp5.emojis import emoji_size, emoji_images, emoji_chars, emoji_names
import matplotlib.pyplot as plt

INPUT_ROWS = 20
INPUT_COLS = 20
INPUT_SIZE = INPUT_COLS * INPUT_ROWS
LATENT_SIZE = 2
HIDDEN_SIZE = 126
HIDDEN_SIZE2 = 258

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
    dataset_input = emoji_images[0:EMOJIS_CHOSEN]
    dataset_input_list = list(dataset_input)

    # set the learning rate and optimizer for training
    optimizer = Adam(1e-2)

    encoder = MLP()
    encoder.addLayer(Dense(inputDim=INPUT_SIZE, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))
    sampler = Sampler(HIDDEN_SIZE, LATENT_SIZE, optimizer=optimizer)

    decoder = MLP()
    decoder.addLayer(Dense(inputDim=LATENT_SIZE, outputDim=HIDDEN_SIZE, activation=ReLU(), optimizer=optimizer))
    decoder.addLayer(Dense(inputDim=HIDDEN_SIZE, outputDim=INPUT_SIZE, activation=Sigmoid(), optimizer=optimizer))

    vae = VAE(encoder, sampler, decoder)

    print(vae)

    my_callbacks = {}  # {"loss": loss_callback}

    vae.train(dataset_input=dataset_input_list, loss=MSE(), metrics=["train_loss", "test_loss"],
              tensorboard=False, epochs=100,
              callbacks=my_callbacks)

    dots = []
    decoder_outputs = []
    amount_correct_characters = 0

    for i in range(len(dataset_input_list)):
        input_reshaped = np.reshape(dataset_input_list[i], (len(dataset_input_list[i]), 1))
        output_history = []
        output = vae.feedforward(input_reshaped, output_history)

        decoder_outputs.append(output)

        dot = (output_history[1][0][0], output_history[1][1][0])
        dots.append(dot)

    # Plot of dataset images
    # Top 20 because SciView has limit of 29 graphs
    for j in range(20):
        graph_fonts(list(dataset_input)[j], decoder_outputs[j])

    # ----------------------
    # Generating new samples
    # ----------------------

    n = 10
    digit_size = INPUT_ROWS
    images = np.zeros((INPUT_ROWS, INPUT_COLS * n))

    random_index1 = np.random.randint(0, len(dataset_input_list))
    input_reshaped1 = np.reshape(dataset_input_list[random_index1], (len(dataset_input_list[random_index1]), 1))
    vae.feedforward(input_reshaped1)
    img1 = vae.sampler.sample

    random_index2 = np.random.randint(0, len(dataset_input_list))
    while random_index1 == random_index2:
        random_index2 = np.random.randint(0, len(dataset_input_list))
    input_reshaped2 = np.reshape(dataset_input_list[random_index2], (len(dataset_input_list[random_index2]), 1))
    vae.feedforward(input_reshaped2)
    img2 = vae.sampler.sample

    for i in range(n):
        z = (img1 * (n - 1 - i) + img2 * i) / (n - 1)
        output = vae.decoder.feedforward(z)
        output = output.reshape(INPUT_ROWS, INPUT_COLS)
        images[:, i * INPUT_COLS:(i + 1) * INPUT_COLS] = output

    plt.figure(figsize=(10, 10))
    plt.title(f"From {emoji_names[random_index1]}({emoji_chars[random_index1]}) "
              f"to {emoji_names[random_index2]}({emoji_chars[random_index2]})")
    plt.imshow(images, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
