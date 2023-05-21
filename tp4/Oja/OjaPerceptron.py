from typing import List

import numpy as np

from tp3.src.theta_funcs import LinealThetaFunction


class OjaPerceptron:
    """Represents a single perceptron, with configurable weights, input size, theta function, and learning rate."""

    def __init__(self, initial_weights: np.ndarray[float]) -> None:
        """Creates a perceptron whose initial weights will be the given vector.
        The length of this vector will be the amount of inputs."""
        self.output = None
        self.h = None
        self.delta_lc_w = None
        self.w = initial_weights
        self.__delta_w = np.zeros(len(initial_weights))
        self.previous_delta_w = np.zeros(len(initial_weights))
        self.theta_func = LinealThetaFunction()

    def evaluate_and_adjust(self, input_data: np.ndarray[float], learning_rate: float) -> float:
        """
        Evaluates the perceptron for a given input, adds weight adjustments, and then returns the result.
        Note that weight adjustments are added to an internal variable and not directly to the weights.
        To apply these adjustments, call update_weights().
        """

        if len(input_data) != len(self.w):
            raise Exception(
                f'Error: during training specified {len(input_data)} inputs to a perceptron with {len(self.w)} '
                f'weights (there should be {len(self.w)} inputs)')

        self.h = self.w @ input_data
        self.output = self.theta_func.primary(self.h)

        # Apply Oja's learning rule
        self.__delta_w += learning_rate * (self.output * input_data - (self.output ** 2) * self.w)
        return self.output

    def update_weights(self) -> None:
        """
        Applies any pending updates to this Perceptron's weights.
        This is done in a separate function and not in evaluate_and_adjust() to make it possible to implement
        both incremental updates (by calling this right after evaluate_and_adjust()) or batch updates
        (by calling only after an epoch finished).
        """
        self.w = self.w + self.__delta_w
        self.__delta_w.fill(0)
