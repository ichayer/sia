from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Activation(ABC):
    _name = "activation name"

    @abstractmethod
    def apply(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def funcFromStr(name):
        functions = {
            "Identity": Identity(),
            "Sigmoid": Sigmoid(),
            "Softmax": Softmax(),
            "ReLU": ReLU(),
            "Tanh": Tanh(),
        }
        return functions[name]

    def _plot(self):
        x = np.linspace(-10, 10, 100)
        plt.plot(x, self.apply(x))
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(self._name)
        plt.show()


class Identity(Activation):
    _name = "Identity function"

    def apply(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)

    def __str__(self):
        return "Identity"


class Sigmoid(Activation):
    _name = "Sigmoid function"

    def apply(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        return self.apply(x) * (1. - self.apply(x))

    def __str__(self):
        return "Sigmoid"


class Tanh(Activation):
    _name = "Tanh - Hyperbolic tangent"

    def apply(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.power(np.tanh(x), 2)

    def __str__(self):
        return "Tanh"


class ReLU(Activation):
    _name = "ReLU - Rectified linear unit"

    def apply(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)

    def __str__(self):
        return "ReLU"


class Softmax(Activation):
    _name = "Softmax function"

    def apply(self, x):
        tmp = np.exp(x - np.max(x, axis=0))  # - np.max(x) prevents under-/overflow
        return tmp / tmp.sum(axis=0)

    def derivative(self, x):
        return self.apply(x) * (1. - self.apply(x))

    def __str__(self):
        return "Softmax"


if __name__ == "__main__":
    for func in [Identity(), Sigmoid(), Tanh(), ReLU(), LeakyReLU(), Softmax()]:
        func._plot()
