from nn import MLP, NoiseFunctionType
from loss import MSE


class Autoencoder(MLP):
    def __init__(self, encoder=MLP(), decoder=MLP(), noise: NoiseFunctionType = None):
        """
        An Autoencoder consists of an Encoder network and a Decoder network.
        In the constructor merged these two networks.
        """
        super().__init__()
        self.layers += encoder.layers + decoder.layers
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise

    def predict(self, input_data):
        """
        Forward propagate the input through the Encoder and
        output the activations of the last layer of the Encoder,
        i.e. return the latent vector.
        """
        return self.encoder.predict(input_data)

    def train(self, dataset_input, dataset_test=None, loss=MSE(), epochs=1, metrics=None, tensorboard=False,
              callbacks=None, batchSize=1):
        if callbacks is None:
            callbacks = {}
        if metrics is None:
            metrics = ["train_loss", "test_loss"]
        super().train(dataset_input, dataset_input, dataset_test, loss=loss, epochs=epochs, metrics=metrics,
                      tensorboard=tensorboard, callbacks=callbacks,
                      autoencoder=True, noise=self.noise, batchSize=batchSize)

    def sampling(self, sampling_coordinates):
        return self.decoder.feedforward(sampling_coordinates)
