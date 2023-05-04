import numpy as np
from perceptron import Perceptron
from trainer import TrainerConfig, train_perceptron, evaluate_perceptron


def run_iteration(
    train_dataset,
    train_dataset_outputs,
    test_dataset,
    test_dataset_outputs,
    config: TrainerConfig,
):
    perceptron = Perceptron(
        initial_weights=np.random.random(len(train_dataset[0]) + 1) * 2 - 1,
        theta_func=config.theta,
    )

    config.print_every = None
    train_perceptron(
        perceptron=perceptron,
        dataset=train_dataset,
        dataset_outputs=train_dataset_outputs,
        config=config,
    )

    training_error = evaluate_perceptron(
        perceptron=perceptron,
        dataset=train_dataset,
        dataset_outputs=train_dataset_outputs,
        error_func=config.error_func,
        print_output=False,
        acceptable_error=config.acceptable_error,
    )

    test_error = evaluate_perceptron(
        perceptron=perceptron,
        dataset=test_dataset,
        dataset_outputs=test_dataset_outputs,
        error_func=config.error_func,
        print_output=False,
        acceptable_error=config.acceptable_error,
    )

    return training_error, test_error


def run_n_times(
    n,
    get_datasets,
    config: TrainerConfig,
):
    training_errors = []
    test_errors = []

    for i in range(n):
        (
            train_dataset,
            train_dataset_outputs,
            test_dataset,
            test_dataset_outputs,
            name
        ) = get_datasets()
        training_error, test_error = run_iteration(
            train_dataset,
            train_dataset_outputs,
            test_dataset,
            test_dataset_outputs,
            config,
        )
        training_errors.append(training_error)
        test_errors.append(test_error)
    return (
        np.mean(training_errors),
        np.std(training_errors),
        np.mean(test_errors),
        np.std(test_errors),
        name
    )
