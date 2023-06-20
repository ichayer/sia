from tp5.optimizer import Adam
from tp5.nn import MLP
from tp5.layers import Dense
from tp5.activations import ReLU, Sigmoid
from tp5.autoencoder import Autoencoder
from tp5.loss import MSE
from tp4.Hopfield.pattern_loader import *
import matplotlib.pyplot as plt

INPUT_SIZE = 35
LATENT_SIZE = 2
HIDDEN_SIZE = 15

NOISE = None

fonts_headers = np.array(
    ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
     "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]
)

new_pattern = np.concatenate([
    ['.', '.', 'X', '.', '.'],
    ['.', 'X', 'X', '.', '.'],
    ['.', '.', 'X', '.', '.'],
    ['.', '.', 'X', '.', '.'],
    ['.', '.', 'X', '.', '.'],
    ['.', '.', 'X', '.', '.'],
    ['.', '.', 'X', '.', '.']
])

pattern_test = [new_pattern]

if __name__ == "__main__":
    dataset_input = load_pattern_map('characters.txt')

    # set the learning rate and optimizer for training
    optimizer = Adam(1e-2)

    encoder = MLP()
    encoder.addLayer(Dense(inputDim=5 * 7, outputDim=15, activation=ReLU(), optimizer=optimizer))

    decoder = MLP()
    decoder.addLayer(Dense(inputDim=15, outputDim=5 * 7, activation=Sigmoid(), optimizer=optimizer))

    autoencoder = Autoencoder(encoder, decoder, noise=NOISE)

    print(autoencoder)

    my_callbacks = {}  # {"loss": loss_callback}

    autoencoder.train(dataset_input=list(dataset_input.values()), dataset_test=pattern_test, loss=MSE(), metrics=["train_loss", "test_loss"], tensorboard=False, epochs=20,
                      callbacks=my_callbacks)

    output = autoencoder.feedforward(list(dataset_input.values())[1])
    plt.imshow(output.reshape(5, 7))
    plt.show()
