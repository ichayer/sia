import numpy as np


def simple(x: float) -> float:
    """A theta function that returns 1 if x >= 0, -1 otherwise."""
    return 1 if x == 0 else np.sign(x)


def lineal(x: float) -> float:
    """An identity function, returns the value unmodified."""
    return x


def tanh(x: float) -> float:
    """A theta function whose image is (-1, 1)."""
    beta = 1
    return np.tanh(beta * x)


def logistic(x: float) -> float:
    """A theta function whose image is (0, 1)."""
    beta = 1
    return 1 / (1 + np.exp(-2*beta*x))
