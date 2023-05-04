import numpy as np
from src.optimizers import *
from src.perceptron import *
from src.trainer import (
    train_multilayer_perceptron,
    TrainerConfig,
    evaluate_multilayer_perceptron,
)

def runner_3b(run_id: str, optimizer: Optimizer, config: TrainerConfig):
    print(f"Run {run_id} started")
    # Abrir el archivo
    with open("digits.txt", "r") as file:
        # Leer cada línea del archivo y convertirla en una lista de enteros
        numbers = [[int(num) for num in line.split()] for line in file]

    # Convertir la lista de listas en un vector de números
    vector = [num for sublist in numbers for num in sublist]

    dataset_input = [vector[i : i + 35] for i in range(0, len(vector), 35)]

    dataset_outputs = [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]]

    perceptrons_by_layer = [35, 10, 10, 1]
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

    multilayer_perceptron_parity = MultilayerPerceptron(perceptrons, optimizer)

    result_parity = train_multilayer_perceptron(
        multilayer_perceptron=multilayer_perceptron_parity,
        dataset=dataset_input[:8],
        dataset_outputs=dataset_outputs[:8],
        config=config,
    )

    avg_err = evaluate_multilayer_perceptron(
        multilayer_perceptron=multilayer_perceptron_parity,
        dataset=dataset_input[8:],
        dataset_outputs=dataset_outputs[8:],
        error_func=config.error_func,
        print_output=True,
        acceptable_error=config.acceptable_error,
    )

    print(f"Run {run_id} finished")
    return {"name": run_id, "data": result_parity, "mean_gen": avg_err}
