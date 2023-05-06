import numpy as np
from src.perceptron import *
from src.trainer import train_multilayer_perceptron, TrainerConfig
from functools import reduce
from src.optimizers import *

dataset = [
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, 1]),
    np.array([-1, -1])
]

dataset_outputs_xor = [[-1], [1], [1], [-1]]

config = TrainerConfig.from_file("ejercicio3-a-config.json")

perceptrons_by_layer = [2, 1]
perceptrons = []

for p in perceptrons_by_layer:
    nl = [0] * p
    perceptrons.append(nl)


for i in range(len(perceptrons_by_layer)):
    for j in range(perceptrons_by_layer[i]):

        if i == 0:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(len(dataset[0]) + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )
        else:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(perceptrons_by_layer[i - 1] + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )

multilayer_perceptron_xor = MultilayerPerceptron(perceptrons, GradientDescent())

result_xor = train_multilayer_perceptron(
    multilayer_perceptron=multilayer_perceptron_xor,
    dataset=dataset,
    dataset_outputs=dataset_outputs_xor,
    config=config
)

print(f"\n\n Epoch: {result_xor.epoch_num}, End Reason: {result_xor.end_reason}, Error: {result_xor.error_history[-1]:.4f}")

