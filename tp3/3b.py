import numpy as np

from tp3.src.optimizers import GradientDescent
from tp3.src.perceptron import *
from tp3.src.trainer import train_multilayer_perceptron, TrainerConfig, evaluate_multilayer_perceptron
from functools import reduce

# Abrir el archivo
with open("digits.txt", "r") as file:
    # Leer cada línea del archivo y convertirla en una lista de enteros
    numbers = [[int(num) for num in line.split()] for line in file]

# Convertir la lista de listas en un vector de números
vector = [num for sublist in numbers for num in sublist]

dataset_input = [vector[i:i+35] for i in range(0, len(vector), 35)]

dataset_outputs = [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]]

config = TrainerConfig.from_file("ejercicio3-b-config.json")

perceptrons_by_layer = [35, 10, 10, 1]
perceptrons = []

for p in perceptrons_by_layer:
    nl = [0] * p
    perceptrons.append(nl)

for i in range(len(perceptrons_by_layer)):
    for j in range(perceptrons_by_layer[i]):

        if i == 0:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(len(dataset_input[0]) + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )
        else:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(perceptrons_by_layer[i - 1] + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )

multilayer_perceptron_parity = MultilayerPerceptron(perceptrons, GradientDescent())

result_parity = train_multilayer_perceptron(
    multilayer_perceptron=multilayer_perceptron_parity,
    dataset=dataset_input[:8],
    dataset_outputs=dataset_outputs[:8],
    config=config
)

print(f"\nEpoch: {result_parity.epoch_num}, End Reason: {result_parity.end_reason}, Error: {result_parity.error_history[-1]:.4f}\n")

print(f"-------Evaluating after training-------\n")
avg_err = evaluate_multilayer_perceptron(
    multilayer_perceptron=multilayer_perceptron_parity,
    dataset=dataset_input[8:],
    dataset_outputs=dataset_outputs[8:],
    error_func=config.error_func,
    print_output=True,
    acceptable_error=config.acceptable_error
)

print(f"\nMultilayer perceptron after training for {result_parity.epoch_num} epoch{''if result_parity.epoch_num == 1 else 's'} has "
      f"an average error of {avg_err} {'✅' if avg_err <=config.acceptable_error else '❌'}\n")