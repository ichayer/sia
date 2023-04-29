import numpy as np

# Error functions take in the expected outputs and actual outputs of a perceptron epoch, and calculate an error value up
# which is up to the interpretation of each function.


def count_nonmatching(expected_outputs: np.ndarray[float], outputs: np.ndarray[float]) -> int:
    """Calculates the error as the amount of elements in the output array which is not exactly the expected value."""
    
    return np.not_equal(expected_outputs, outputs).sum()


def cost_average(expected_outputs: np.ndarray[float], outputs: np.ndarray[float]) -> float:
    """Calculates the error as the sum of the costs of each element, which is (expected - output)**2 / 2"""
    
    return np.average(np.power(expected_outputs - outputs, 2)) * 0.5


def cost_max(expected_outputs: np.ndarray[float], outputs: np.ndarray[float]) -> float:
    """Calculates the error as the sum of the costs of each element, which is (expected - output)**2 / 2"""
    
    return np.max(np.power(expected_outputs - outputs, 2)) * 0.5


map = {
    "count_nonmatching": count_nonmatching,
    "cost_average": cost_average,
    "cost_max": cost_max
}