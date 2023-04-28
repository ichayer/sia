import numpy as np
from matplotlib import pyplot as plt
from .perceptron import Perceptron

def plot2d(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float]):
    colors = []
    for i in range(len(dataset)):
        if dataset_outputs[i] < 0:
            colors.append("black" if perceptron.evaluate(dataset[i]) < 0 else "gray")
        else:
            colors.append("red" if perceptron.evaluate(dataset[i]) >= 0 else "orange")
    
    weights = perceptron.w
    if (weights[1] < weights[2]):
        line_origin = (0, -weights[0] / weights[2])
    else:
        line_origin = (weights[0] / weights[1], 0)
    line_advance = (line_origin[0] - weights[2], line_origin[1] + weights[1])
    
    x, y = np.asarray(dataset).T
    plt.scatter(x, y, c=colors)
    plt.axline(line_origin, line_advance)
    plt.axis('equal')
    return plt.show()
