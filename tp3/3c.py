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
prob_range = [0.05, 0.15, 0.25]
num_runs = 10
performance_means = []
performance_stds = []

for (m, prob) in enumerate(prob_range):
    accuracy_results = []
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

        multilayer_perceptron_number = MultilayerPerceptron(perceptrons, Momentum())

        result_number = train_multilayer_perceptron(
            multilayer_perceptron=multilayer_perceptron_number,
            dataset=flat_dataset_input[:train_items],
            dataset_outputs=dataset_outputs[:train_items],
            config=config
        )

        print(
            f"\nEpoch: {result_number.epoch_num}, End Reason: {result_number.end_reason}, Error: {result_number.error_history[-1]:.4f}\n")

        # Sigo entrenandolo con imagenes con ruido (30 imagenes con add line noise 0.2)

        print("----- Original Images -----\n")
        print_images(images)

        noisy_images = []
        iters = 3
        for g in range(iters):
            for image in images:
                noisy_images.append(add_line_noise(np.array(image), 0.2, 0.2))

        noisy_flat_dataset_input = []
        for lista in noisy_images:
            flat_dataset_input.append([elem for sublista in lista for elem in sublista])

        result_number = train_multilayer_perceptron(
            multilayer_perceptron=multilayer_perceptron_number,
            dataset=noisy_flat_dataset_input,
            dataset_outputs=np.tile(dataset_outputs, iters),
            config=config
        )

        # Generalizacion

        noisy_images = []
        for image in images:
            noisy_images.append(salt_and_pepper_noise(np.array(image), prob, prob))

        flat_dataset_input = []
        for lista in noisy_images:
            flat_dataset_input.append([elem for sublista in lista for elem in sublista])

        print(f"----- Noise Line Images with {prob} probability-----\n")
        print_images(noisy_images)

        print(f"----- Evaluating after training -----\n")
        evaluation_results = evaluate_multilayer_perceptron(
            multilayer_perceptron=multilayer_perceptron_number,
            dataset=flat_dataset_input[:train_items],
            dataset_outputs=dataset_outputs[:train_items],
            print_output=False,
        )

        network_errors = evaluation_results["err"]
        network_expected = evaluation_results["expected_output"]
        network_output = evaluation_results["network_output"]

        accuracy = 0
        for (j, neuron_output) in enumerate(network_output):
            max_ind = np.argmax(neuron_output)
            if j == max_ind:
                accuracy += 1
            print(f"For the image of the {j} digit, the perceptron interpreted a {max_ind}")
        accuracy = accuracy / len(images)
        accuracy_results.append(accuracy)
    performance_means.append(np.mean(accuracy_results))
    performance_stds.append(np.std(accuracy_results))

plt.bar(prob_range, performance_means, yerr=performance_stds, capsize=5, width=0.05)

plt.xlabel('H|V noise lines probability')
plt.ylabel('Accuracy percentage')
plt.title('Perceptron performance adding noise lines')
plt.show()
