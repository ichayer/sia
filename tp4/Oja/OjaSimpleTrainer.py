import numpy as np

from tp4.Oja.OjaPerceptron import OjaPerceptron


class OjaSimpleTrainer:
    """Trains a single OjaPerceptron with a given dataset."""

    def __init__(self, perceptron: OjaPerceptron, inputs: dict[np.ndarray[float]]):
        self.learning_rate = 0.1
        self.max_epochs = 30
        self.current_epoch = 0
        self.perceptron = perceptron
        self.inputs = inputs
        self.print_every = 10

    def train(self) -> None:
        while self.current_epoch < self.max_epochs:
            self.current_epoch += 1
            for input_data in self.inputs.values():
                self.perceptron.evaluate_and_adjust(input_data, self.learning_rate)
                self.perceptron.update_weights()
            if self.current_epoch % self.print_every == 0:
                print(f"Epoch {self.current_epoch} finished.")
                print(f"Current weights: {self.perceptron.w}")
