import numpy as np
from matplotlib import pyplot as plt
from .perceptron import Perceptron


def plot_points2d(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float]) -> plt:
    colors = []
    for i in range(len(dataset)):
        if dataset_outputs[i] < 0:
            colors.append("black" if perceptron.evaluate(dataset[i]) < 0 else "gray")
        else:
            colors.append("red" if perceptron.evaluate(dataset[i]) >= 0 else "orange")
    
    x, y = np.asarray(dataset).T
    plt.scatter(x, y, c=colors)
    plt.axis('equal')
    return plt



def plot2d(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float]) -> plt:
    plot_points2d(perceptron, dataset, dataset_outputs)

    weights = perceptron.w
    if (weights[1] < weights[2]):
        line_origin = (0, -weights[0] / weights[2])
    else:
        line_origin = (weights[0] / weights[1], 0)
    line_advance = (line_origin[0] - weights[2], line_origin[1] + weights[1])

    plt.axline(line_origin, line_advance)
    return plt
