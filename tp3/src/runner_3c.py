import numpy as np

from tp3.src.optimizers import *
from tp3.src.perceptron import *
from tp3.src.trainer import (
    train_multilayer_perceptron,
    TrainerConfig,
    evaluate_multilayer_perceptron,
    EndReason,
)


def run_by_optimizer(
    run_id: str,
    optimizer: Optimizer,
    config: TrainerConfig,
):
    print(f"Run {run_id} started")
    # Input
    with open("../digits.txt", "r") as file:
        numbers = [[int(num) for num in line.split()] for line in file]

    vector = [num for sublist in numbers for num in sublist]
    dataset_input = [vector[i : i + 35] for i in range(0, len(vector), 35)]

    # Input with noise
    with open("../digits_noise.txt", "r") as file:
        numbers = [[int(num) for num in line.split()] for line in file]

    vector = [num for sublist in numbers for num in sublist]
    dataset_input_noise = [vector[i : i + 35] for i in range(0, len(vector), 35)]

    # Output
    dataset_outputs = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    for i in range(len(dataset_outputs)):
        for j in range(len(dataset_outputs[i])):
            if dataset_outputs[i][j] == 0:
                dataset_outputs[i][j] = -1

    perceptrons_by_layer = [35, 10, 10, 10]
    perceptrons = []

    for p in perceptrons_by_layer:
        nl = [0] * p
        perceptrons.append(nl)

    for i in range(len(perceptrons_by_layer)):
        for j in range(perceptrons_by_layer[i]):
            if i == 0:
                perceptrons[i][j] = Perceptron(
                    initial_weights=np.random.random(len(dataset_input[0]) + 1) * 0.8
                    - 0.4,
                    theta_func=config.theta,
                )
            else:
                perceptrons[i][j] = Perceptron(
                    initial_weights=np.random.random(perceptrons_by_layer[i - 1] + 1)
                    * 0.8
                    - 0.4,
                    theta_func=config.theta,
                )

    multilayer_perceptron_number = MultilayerPerceptron(perceptrons, optimizer)

    n_train_items = 10

    result_number = train_multilayer_perceptron(
        multilayer_perceptron=multilayer_perceptron_number,
        dataset=dataset_input[:n_train_items],
        dataset_outputs=dataset_outputs[:n_train_items],
        config=config,
    )
    # Generalization

    avg = evaluate_multilayer_perceptron(
        multilayer_perceptron=multilayer_perceptron_number,
        dataset=dataset_input[:n_train_items],
        dataset_outputs=dataset_outputs[:n_train_items],
        print_output=False,
    )

    print(f"Run {run_id} finished")
    return {
        "name": run_id,
        "data": result_number,
        "mean_gen": avg,
        "has_converged": result_number.end_reason == EndReason.ACCEPTABLE_ERROR_REACHED,
    }
