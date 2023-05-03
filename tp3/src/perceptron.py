import numpy as np
from tp3.src.theta_funcs import ThetaFunction


class Perceptron:
    """Represents a single perceptron, with configurable weights, input size, theta function, and learning rate."""

    def __init__(self, initial_weights: np.ndarray[float], theta_func: ThetaFunction) -> None:
        """Creates a perceptron whose initial weights will be the given vector. The length of this vector will be the amount of inputs."""
        self.output = None
        self.h = None
        self.delta_lc_w = None
        self.w = initial_weights
        self.__delta_w = np.zeros(len(initial_weights))
        self.theta_func = theta_func

    def evaluate(self, input_data: np.ndarray[float]) -> float:
        """Evaluates the perceptron for a given input and returns the result."""
        if len(input_data) != len(self.w) - 1:
            raise Exception(
                f'Error: specified {len(input_data)} inputs to a perceptron with {len(self.w)} weights (there should be {len(self.w) - 1} inputs)')

        self.h = self.w[1:] @ input_data + self.w[0]
        self.output = self.theta_func.primary(self.h)
        return self.output

    def evaluate_and_adjust(self, input_with_one: np.ndarray[float], expected_output: float,
                            learning_rate: float) -> float:
        """
        Evaluates the perceptron for a given input, adds weight adjustments, and then returns the result. Note that weight adjustments are added to
        an internal variable and not directly to the weights. To apply these adjustments, call update_weights(). The input must be prepended with a 1.
        """
        if len(input_with_one) != len(self.w):
            raise Exception(
                f'Error: during training specified {len(input_with_one)} inputs to a perceptron with {len(self.w)} weights (there should be {len(self.w)} inputs, did you prepend the input with a 1?)')

        self.h = self.w @ input_with_one
        self.output = self.theta_func.primary(self.h)

        if self.output != expected_output:
            # self.__delta_w += 2 * learning_rate * expected_output * input
            self.__delta_w += learning_rate * (expected_output - self.output) * self.theta_func.derivative(self.output, self.h) * input_with_one
        return self.output

    def adjust(self, input_data: np.ndarray[float], delta_lc_w: float, learning_rate: float) -> None:
        self.delta_lc_w = delta_lc_w
        for i in range(len(self.__delta_w)):
            if i == 0:
                self.__delta_w[i] += learning_rate * delta_lc_w
            else:
                self.__delta_w[i] += learning_rate * delta_lc_w * input_data[i - 1]

    def update_weights(self) -> None:
        """
        Applies any pending updates to this perceptron's weights. This is done in a separate function and not in evaluate_and_adjust() to make it
        possible to implement both incremental updates (by calling this right after evaluate_and_adjust()) or batch updates (by calling only after
        an epoch finished).
        """
        self.w = self.w + self.__delta_w
        self.__delta_w.fill(0)


class MultilayerPerceptron:

    def __init__(self, perceptron_layers: list[list[Perceptron]]) -> None:

        if perceptron_layers is None or len(perceptron_layers) < 2:
            raise ValueError("Multilayer perceptron must have at least 2 layers")

        for i in range(len(perceptron_layers) - 1):
            for perceptron in perceptron_layers[i + 1]:
                if (len(perceptron.w) - 1) != len(perceptron_layers[i]):
                    raise ValueError("Invalid perceptron structure")

        self.perceptron_layers = perceptron_layers
        self.total_layers = len(perceptron_layers)
        self.last_layer = perceptron_layers[-1]
        self.results = [[0.0] * len(sublist) for sublist in self.perceptron_layers]

    def __feed_forward(self, input_data: np.ndarray[float]) -> None:
        for (i, perceptron) in enumerate(self.perceptron_layers[0]):
            self.results[0][i] = perceptron.evaluate(input_data)

        for (i, sublist) in enumerate(self.perceptron_layers[1:], start=1):
            for j, perceptron in enumerate(sublist):
                self.results[i][j] = perceptron.evaluate(self.results[i - 1])

    def evaluate_and_adjust(self, input_data: np.ndarray[float], expected_output: list[float],
                            learning_rate: float) -> None:

        self.__feed_forward(input_data)
        # Backpropagation
        # for i in range(len(self.perceptron_layers)-1, -1, -1):
        #     for (j, perceptron) in enumerate(self.perceptron_layers[i]):
        #         # TODO: cambiar a optimizer
        #         if i!=0:
        #             self.delta_w[i][j] = gradient_desc(self.perceptron_weights[i][j], self.delta_w[i][j], perceptron.theta_func, self.results[i-1], learning_rate)
        #         else:
        #             self.delta_w[i][j] = gradient_desc(self.perceptron_weights[i][j], self.delta_w[i][j], perceptron.theta_func, input, learning_rate)

        for (i, perceptron) in enumerate(self.perceptron_layers[-1]):
            delta_lc_w = (expected_output[i] - perceptron.output) * perceptron.theta_func.derivative(perceptron.output, perceptron.h)
            perceptron.adjust(self.results[-2], delta_lc_w, learning_rate)

        for i in range(len(self.perceptron_layers) - 2, -1, -1):
            for j, perceptron in enumerate(self.perceptron_layers[i]):
                delta_lc_w = 0
                for perceptron_parent in self.perceptron_layers[i + 1]:
                    delta_lc_w += perceptron_parent.delta_lc_w * perceptron_parent.w[j]
                delta_lc_w * perceptron.theta_func.derivative(perceptron.output, perceptron.h)

                if i != 0:
                    perceptron.adjust(self.results[i - 1], delta_lc_w, learning_rate)
                else:
                    perceptron.adjust(input_data, delta_lc_w, learning_rate)

    def update_weights(self):
        for sublist in self.perceptron_layers:
            for perceptron in sublist:
                perceptron.update_weights()
