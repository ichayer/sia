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
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        aux = np.zeros(len(perceptron_dw))
        for i in range(len(perceptron_dw)):
            aux[i] += self.alpha * perceptron.previous_delta_w[i] + learning_rate * gt[i]
        return aux

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = self.m = self.v = 0

    def adjust(self, perceptron, perceptron_dw, gt, learning_rate):
        aux = np.zeros(len(perceptron_dw))
        self.t += 1
        for i in range(len(perceptron_dw)):
            self.m = self.beta1 * self.m + (1 - self.beta1) * gt[i]
            self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(gt[i], 2)
            m_mean = self.m / (1 - np.power(self.beta1, self.t))
            v_mean = self.v / (1 - np.power(self.beta2, self.t))
            aux[i] += learning_rate * m_mean / (np.sqrt(v_mean) + self.epsilon)
        return aux