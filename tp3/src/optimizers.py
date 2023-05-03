import numpy as np

class Optimizer:
    def __init__(self):
        pass

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self):
        super().__init__()

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        aux = np.zeros(len(perceptron_dw))
        for i in range(len(perceptron_dw)):
            aux[i] += learning_rate * gt[i]
        return aux

class Momentum(Optimizer):
    def __init__(self, beta1=0.9, alpha=0.8):
        super().__init__()
        self.beta1 = beta1
        self.alpha= alpha

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        # Add self.__previous_delta_w = np.zeros(len(initial_weights)) in Perceptron
        super().adjust(perceptron, perceptron_dw, gt, learning_rate)

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.epsilon = epsilon

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        super().adjust(perceptron, perceptron_dw, gt, learning_rate)