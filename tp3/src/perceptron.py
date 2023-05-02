import numpy as np
from .theta_funcs import ThetaFunction


class Perceptron:
    """Represents a single perceptron, with configurable weights, input size, theta function, and learning rate."""
    
    def __init__(self, initial_weights: np.ndarray[float], theta_func: ThetaFunction) -> None:
        """Creates a perceptron whose initial weights will be the given vector. The length of this vector will be the amount of inputs."""
        self.w = initial_weights
        self.__delta_w = np.zeros(len(initial_weights))
        self.theta_func = theta_func
    
    def evaluate(self, input: np.ndarray[float]) -> float:
        """Evaluates the perceptron for a given input and returns the result."""
        if len(input) != len(self.w) - 1:
            raise Exception(f'Error: specified {len(input)} inputs to a perceptron with {len(self.w)} weights (there should be {len(self.w) - 1} inputs)')
        
        h = self.w[1:] @ input + self.w[0]
        return self.theta_func.primary(h)
    
    def evaluate_and_adjust(self, input_with_one: np.ndarray[float], expected_output: float, learning_rate: float) -> float:
        """
        Evaluates the perceptron for a given input, adds weight adjustments, and then returns the result. Note that weight adjustments are added to
        an internal variable and not directly to the weights. To apply these adjustments, call update_weights(). The input must be prepended with a 1.
        """
        if len(input_with_one) != len(self.w):
            raise Exception(f'Error: during training specified {len(input_with_one)} inputs to a perceptron with {len(self.w)} weights (there should be {len(self.w)} inputs, did you prepend the input with a 1?)')
        
        h = self.w @ input_with_one
        output = self.theta_func.primary(h)
    
        if output != expected_output:
            # self.__delta_w += 2 * learning_rate * expected_output * input
            self.__delta_w += learning_rate * (expected_output - output) * self.theta_func.derivative(output, h) * input_with_one
        
        return output
    
    def update_weights(self) -> None:
        """
        Applies any pending updates to this perceptron's weights. This is done in a separate function and not in evaluate_and_adjust() to make it
        possible to implement both incremental updates (by calling this right after evaluate_and_adjust()) or batch updates (by calling only after
        an epoch finished).
        """
        self.w = self.w + self.__delta_w
        self.__delta_w.fill(0)

class MultilayerPerceptron:

    _LAST_LAYER_NEURON_DEFAULT_INDEX = 1
    def __init__(self, perceptron_layers: list[list[Perceptron]], inputs_quantity=1) -> None:
        self.total_layers = len(perceptron_layers) + 1  # The reason to add 1 is due to the initial layer we are not counting.
        self.perceptron_layers = perceptron_layers

        # Create weights for the initial layer
        initial_layer_weights = np.random.randn(perceptron_layers[0], inputs_quantity)
        self.perceptron_weights = [initial_layer_weights]

        # Create weights for the rest of the layers
        for layer in perceptron_layers:
            layer_weights = []
            for perceptron in layer:
                layer_weights.append(perceptron.w)
            self.perceptron_weights.append(np.array(layer_weights))

        # Create delta_w for the rest of the layers based on perceptron weights
        self.__delta_w = []
        for pw in self.perceptron_weights:
            delta_w = np.zeros_like(pw)
            self.__delta_w.append(delta_w)

        self.last_layer_neuron_idx = self._LAST_LAYER_NEURON_DEFAULT_INDEX

    def update_neuron_idx(self, idx):
        if idx < 0 or idx > len(self.perceptron_layers[-1]):
            raise Exception(f'Error: invalid neuron index {idx} for last layer with {len(self.perceptron_layers[-1])} neurons')
        self.last_layer_neuron_idx = idx

    def evaluate(self, input_data: np.ndarray[float]) -> float:
        """
        Computes the output of the multilayer perceptron for the given input x.
        """
        pass

    def evaluate_and_adjust(self, input_with_one, expected_output, learning_rate):
        """
        Evaluates the MLP for a given input, adds weight adjustments, and then returns the result. Note that weight adjustments are added to
        an internal variable and not directly to the weights. To apply these adjustments, call update_weights(). The input must be prepended with a 1.
        """
        pass

    def update_weights(self):
        """
        Applies any pending updates to this MLP's weights. This is done in a separate function and not in evaluate_and_adjust() to make it
        possible to implement both incremental updates (by calling this right after evaluate_and_adjust()) or batch updates (by calling only after
        an epoch finished).
        """
        for i in range(1, self.total_layers):
            self.perceptron_weights[i] = self.perceptron_weights[i] + self.__delta_w[i]
            self.__delta_w[i].fill(0)



