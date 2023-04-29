import numpy as np
from abc import ABC, abstractmethod


class ThetaFunction(ABC):
    @abstractmethod
    def primary(self, x: float) -> float:
        """Used to transform the output of a perceptron."""
        pass

    @abstractmethod
    def derivative(self, p: float, x: float) -> float:
        """Used to multiply the delta_w of a perceptron while training."""
        pass


class SimpleThetaFunction(ThetaFunction):
    """A theta function that returns 1 if x >= 0, -1 otherwise."""

    def __init__(self, config) -> None:
        pass

    def primary(self, x: float) -> float:
        return 1 if x == 0 else np.sign(x)

    def derivative(self, p: float, x: float) -> float:
        return 1


class LinealThetaFunction(ThetaFunction):
    """An identity function, returns the value unmodified."""

    def __init__(self, config) -> None:
        pass

    def primary(self, x: float) -> float:
        return x

    def derivative(self, p: float, x: float) -> float:
        return 1


class TanhThetaFunction(ThetaFunction):
    """A theta function whose image is (-1, 1) by default."""

    def __init__(self, config) -> None:
        self.beta = float(config['beta'])
        if 'range' in config:
            range = config['range']
            range0 = float(range[0])
            range1 = float(range[1])
            if range0 >= range1:
                raise Exception(f"Invalid range: {range}. Must be an array of two elements in ascending order.")
            t = (range1 - range0) / 2
            self.multiplier = t
            self.additive = t + range0
        else:
            self.multiplier = 1
            self.additive = 0

    def primary(self, x: float) -> float:
        return np.tanh(self.beta * x) * self.multiplier + self.additive

    def derivative(self, p: float, x: float) -> float:
        return self.beta * (1 - p*p) * self.multiplier


class LogisticThetaFunction(ThetaFunction):
    """A logistic function whose image is (0, 1) by default."""

    def __init__(self, config) -> None:
        self.beta = float(config['beta'])
        if 'range' in config:
            range = config['range']
            range0 = float(range[0])
            range1 = float(range[1])
            if range0 >= range1:
                raise Exception(f"Invalid range: {range}. Must be an array of two elements in ascending order.")
            self.multiplier = range1 - range0
            self.additive = range0
        else:
            self.multiplier = 1
            self.additive = 0


    def primary(self, x: float) -> float:
        return 1 / (1 + np.exp(-2 * self.beta * x)) * self.multiplier + self.additive

    def derivative(self, p: float, x: float) -> float:
        return 2 * self.beta * p * (1 - p) * self.multiplier


map = {
    "simple": SimpleThetaFunction,
    "lineal": LinealThetaFunction,
    "tanh": TanhThetaFunction,
    "logistic": LogisticThetaFunction
}