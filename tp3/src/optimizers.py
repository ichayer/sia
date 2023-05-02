import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.m = None

    def update_weights(self, weights, delta_w):
        pass

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.1, beta1=0.9):
        super().__init__(learning_rate)
        self.beta1 = beta1

    def update_weights(self, weights, delta_w):
        self.m = [np.zeros_like(w) for w in weights]

        for i in range(len(weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta_w[i]
            weights[i] -= self.learning_rate * self.m[i]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.v = None
        self.epsilon = epsilon

    def update_weights(self, weights, delta_w):
        self.m = [np.zeros_like(w) for w in weights]
        self.v = [np.zeros_like(w) for w in weights]
        self.t += 1

        alpha = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for i in range(len(weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta_w[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * delta_w[i] ** 2
            weights[i] -= alpha * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)