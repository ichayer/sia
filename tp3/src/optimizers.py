import numpy as np


from typing import List
from .theta_funcs import ThetaFunction

def momentum(weights: List[np.ndarray], delta_w: List[np.ndarray], learning_rate: float=0.1, beta1: float=0.9) -> List[np.ndarray]:
    m = [np.zeros_like(w) for w in weights]

    for i in range(len(weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * delta_w[i]
        weights[i] -= learning_rate * m[i]

    return weights

def gradient_desc(perceptron_weights: list[float], perceptron_delta_w: float, theta_fun : ThetaFunction, results: list[float], learning_rate: float=0.1) -> float:
    
    
    
    
    
    for i in range(len(perceptron_weights)):
        perceptron_weights[i] -= learning_rate * perceptron_delta_w[i] 
    return perceptron_weights


def adam(t : int, weights: List[np.ndarray], delta_w: List[np.ndarray], learning_rate: float=0.1, beta1: float=0.9, beta2 : float=0.999, epsilon: float=1e-8) -> List[np.ndarray]:
    m = [np.zeros_like(w) for w in weights]
    v = [np.zeros_like(w) for w in weights]

    alpha = learning_rate * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for i in range(len(weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * delta_w[i]
        v[i] = beta2 * v[i] + (1 - beta2) * delta_w[i]**2
        weights[i] -= alpha * m[i] / (np.sqrt(v[i]) + epsilon)
    
    return weights

optimizer_list = {
    "momentum": momentum,
    "gradient_desc": gradient_desc,
    "adam": adam
}


def optimizer(config_optimizer: str, weights: list[float]) -> int:
    return optimizer_list[config_optimizer]




