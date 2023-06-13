import numpy as np

class Optimizer:
    def __init__(self):
        pass

    def adjust(self, perceptron, gt, learning_rate):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self):
        super().__init__()

    def adjust(self, perceptron, gt, learning_rate):
        return learning_rate * gt


class Momentum(Optimizer):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def adjust(self, perceptron, gt, learning_rate):
        return learning_rate * gt - self.alpha * perceptron.previous_delta_w


class Adam(Optimizer):
    def __init__(self, b1=0.9, b2=0.999):
        super().__init__()
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2
        self.t = 0

    def adjust(self, perceptron, gt, learning_rate):
        # If not initialized
        self.t += 1
        if self.m is None:
            self.m = np.zeros(np.shape(gt))
            self.v = np.zeros(np.shape(gt))

        self.m = self.b1 * self.m + (1 - self.b1) * gt
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(gt, 2)

        m_hat = self.m / (1 - np.power(self.b1, self.t))
        v_hat = self.v / (1 - np.power(self.b2, self.t))

        w_updt = learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        # return w - self.w_updt
        return w_updt
