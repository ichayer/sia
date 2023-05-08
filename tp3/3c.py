import numpy as np
from colr import color
import random

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
layers = [35, 10, 10, 10]

perceptrons = []
for p in layers:
    nl = [0] * p
    perceptrons.append(nl)

for i in range(len(layers)):
    for j in range(layers[i]):

        if i == 0:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(len(flat_dataset_input[0]) + 1) * 2 * weight_extreme - weight_extreme,
                theta_func=config.theta
            )
        else:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(layers[i - 1] + 1) * 2 * weight_extreme - weight_extreme,
                theta_func=config.theta
            )

multilayer_perceptron_number = MultilayerPerceptron(perceptrons, Momentum())


result_number = train_multilayer_perceptron(
    multilayer_perceptron=multilayer_perceptron_number,
    dataset=flat_dataset_input[:train_items],
    dataset_outputs=dataset_outputs[:train_items],
    config=config
)

print(f"\nEpoch: {result_number.epoch_num}, End Reason: {result_number.end_reason}, Error: {result_number.error_history[-1]:.4f}\n")

# Generalization

print("----- Original Images -----\n")
print_images(images)

# Salt and Pepper
noisy_images = []
for image in images:
    noisy_images.append(salt_and_pepper_noise(np.array(image), 0.2, 0.2))
    
flat_dataset_input = []
for lista in noisy_images:
    flat_dataset_input.append([elem for sublista in lista for elem in sublista])

print("----- Salt and Pepper Images -----\n")
print_images(noisy_images)

print(f"----- Evaluating after training -----\n")
evaluation_results = evaluate_multilayer_perceptron(
    multilayer_perceptron=multilayer_perceptron_number,
    dataset=flat_dataset_input[:train_items],
    dataset_outputs=dataset_outputs[:train_items],
    print_output=True,
    acceptable_error=0.1,
    error_func=config.error_func
)

network_errors = evaluation_results["err"]
network_expected = evaluation_results["expected_output"]
network_output = evaluation_results["network_output"]

for (i, neuron_output) in enumerate(network_output):
	max_ind = np.argmax(neuron_output)
	print(f"Para la imagen del {i}, el perceptrón interpretó un {max_ind}")