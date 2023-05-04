import numpy as np
from .perceptron import *
from .trainer import train_multilayer_perceptron, TrainerConfig, EndReason
from functools import reduce
from .optimizers import *

def run_by_layers(run_id ,perceptrons_by_layer, config: TrainerConfig):
    print(f"Run {run_id} started")
    dataset = [
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([-1, 1]),
        np.array([-1, -1])
    ]

    dataset_outputs_xor = [[-1], [1], [1], [-1]]
    
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

    print(f"Run {run_id} finished")
    return {
        "name": run_id,
        "data": result_xor,
        "epochs": result_xor.epoch_num,
        "has_converged": result_xor.end_reason == EndReason.ACCEPTABLE_ERROR_REACHED,
    }
