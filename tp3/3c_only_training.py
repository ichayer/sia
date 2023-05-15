import numpy as np
from colr import color
import random
import matplotlib.pyplot as plt

from src.optimizers import *
from src.perceptron import *
from src.noise_funcs import *
from src.trainer import train_multilayer_perceptron, TrainerConfig, evaluate_multilayer_perceptron

# Input
with open('digits.txt', "r") as file:
    numbers = [[int(num) for num in line.split()] for line in file]
images = []
image = []
for i, sublist in enumerate(numbers):
    if i % 7 == 0 and i != 0:
        images.append(image)
        image = []
    if sublist:
        image.append(sublist)
images.append(image)

flat_dataset_input = []
for lista in images:
    flat_dataset_input.append([elem for sublista in lista for elem in sublista])

# Output
dataset_outputs = []
for i in range(10):
    l = [-1] * 10
    l[i] = 1
    dataset_outputs.append(l)

# Config
config = TrainerConfig.from_file("ejercicio3-c-config.json")
train_items = 10
weight_extreme = 0.4,
layers = [35, 15, 10]
methods = ["10^-1", "10^-2", "10^-3"]
num_runs = 2
average_errors = []
errors = []

for (m, method) in enumerate(methods):
    errors.append([])
    average_errors.append([])
    for k in range(num_runs):
        perceptrons = []
        for p in layers:
            nl = [0] * p
            perceptrons.append(nl)

        for i in range(len(layers)):
            for j in range(layers[i]):
                if i == 0:
                    perceptrons[i][j] = Perceptron(
                        initial_weights=np.random.random(
                            len(flat_dataset_input[0]) + 1) * 2 * weight_extreme - weight_extreme,
                        theta_func=config.theta
                    )
                else:
                    perceptrons[i][j] = Perceptron(
                        initial_weights=np.random.random(layers[i - 1] + 1) * 2 * weight_extreme - weight_extreme,
                        theta_func=config.theta
                    )

        multilayer_perceptron_number = MultilayerPerceptron(perceptrons, GradientDescent())

        config.learning_rate = 1/np.power(10, m+1)
        result_number = train_multilayer_perceptron(
            multilayer_perceptron=multilayer_perceptron_number,
            dataset=flat_dataset_input[:train_items],
            dataset_outputs=dataset_outputs[:train_items],
            config=config
        )

        errors[-1].append(result_number.error_history)
        print(len(errors[-1]))

    average_errors[-1].append(np.array([np.mean(pos) for pos in zip(*errors[-1])]))
    print(f"Average errors: {m}")
    print(average_errors[m][0])
    print(len(average_errors[m][0]))

plt.plot(list(range(1, 2501)), average_errors[0][0], color='red', label='10^-1')
plt.plot(list(range(1, 2501)), average_errors[1][0], color='green', label='10^-2')
plt.plot(list(range(1, 2501)), average_errors[2][0], color='blue', label='10^-3')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error through epochs with 100% of the dataset')

# Mostrar el gráfico
plt.show()

