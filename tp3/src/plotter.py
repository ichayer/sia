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
        line_origin = (-weights[0] / weights[1], 0)
    line_advance = (line_origin[0] - weights[2], line_origin[1] + weights[1])

    plt.axline(line_origin, line_advance)
    return plt

def plot_arrays(arrays, arrays_labels, title, xlabel, ylabel) -> plt:
    colors = ['r', 'g', 'b']

    for i in range(len(arrays)):
        values = arrays[i]
        label = arrays_labels[i]
        # Create an array of numbers
        numbers = np.array(values)
        # Create a corresponding array for the x-axis (indices of the numbers)
        x_axis = np.arange(len(numbers))
        # Plot the numbers
        plt.plot(x_axis, numbers, linestyle='-', color=colors[i], label=label, linewidth=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.legend()

    return plt