import json

import numpy as np

from tp4.Oja.OjaPerceptron import OjaPerceptron


class Config:
    """Configuration for OjaSimpleTrainer."""

    def __init__(self, learning_rate: float, max_epochs: int, print_every: int):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.print_every = print_every
        self.should_print = print_every > 0

    @staticmethod
    def load_from_file(path: str) -> 'Config':
        with open(path, 'r') as config_file:
            config_lines = json.load(config_file)
            return Config(config_lines['learning_rate'], config_lines['max_epochs'], config_lines['print_every'])


class OjaSimpleTrainer:
    """Trains a single OjaPerceptron with a given dataset."""

    def __init__(self, perceptron: OjaPerceptron, inputs: dict[np.ndarray[float]], config: Config):
        self.learning_rate = config.learning_rate
        self.max_epochs = config.max_epochs
        self.print_every = config.print_every
        self.should_print = config.should_print
        self.perceptron = perceptron
        self.inputs = inputs
        self.current_epoch = 0

    def train(self) -> None:
        while self.current_epoch < self.max_epochs:
            self.current_epoch += 1
            for input_data in self.inputs.values():
                self.perceptron.evaluate_and_adjust(input_data, self.learning_rate)
                self.perceptron.update_weights()
            if self.should_print and (self.current_epoch % self.print_every == 0):
                # Print current epoch, weights, and inputs labels
                print(f"Epoch {self.current_epoch} finished.")
                print(f"Current weights: {self.perceptron.w}")
