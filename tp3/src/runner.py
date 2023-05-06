import numpy as np
from .perceptron import Perceptron
from .trainer import TrainerConfig, train_perceptron, evaluate_perceptron, TrainerResult, Scaler

def normalize_errors(errors: list[float], scaler: Scaler) -> list[float]:
    return [scaler.reverse(error) for error in errors]  

class IterationResult:
    def __init__(
        self,
        training_error: float,
        test_error: float,
        result: TrainerResult,
        test_error_history: list[float],
        test_error_history_normalized: list[float],
    ):
        self.training_error = training_error
        self.test_error = test_error
        self.result = result
        self.test_error_history = test_error_history
        self.test_error_history_normalized = test_error_history_normalized


def run_iteration(
    train_dataset,
    train_dataset_outputs,
    test_dataset,
    test_dataset_outputs,
    config: TrainerConfig,
) -> IterationResult:
    test_error_history_normalized = []

    initial_weights = np.zeros(len(train_dataset[0]) + 1)

    perceptron = Perceptron(
        initial_weights=initial_weights,
        theta_func=config.theta,
    )

    config.print_every = None
    result = train_perceptron(
        perceptron=perceptron,
        dataset=train_dataset,
        dataset_outputs=train_dataset_outputs,
        test_dataset=test_dataset,
        test_dataset_outputs=test_dataset_outputs,
        config=config,
    )

    training_error = evaluate_perceptron(
        perceptron=perceptron,
        dataset=train_dataset,
        dataset_outputs=train_dataset_outputs,
        error_func=config.error_func,
        scaler=config.scaler,
        print_output=False,
        acceptable_error=config.acceptable_error,
    )

    test_error = evaluate_perceptron(
        perceptron=perceptron,
        dataset=test_dataset,
        dataset_outputs=test_dataset_outputs,
        error_func=config.error_func,
        scaler=config.scaler,
        print_output=False,
        acceptable_error=config.acceptable_error,
    )

    test_error_history_normalized = normalize_errors(result.test_error_history, config.scaler)
    

    return IterationResult(
        training_error=training_error,
        test_error=test_error,
        result=result,
        test_error_history=result.test_error_history,
        test_error_history_normalized=test_error_history_normalized,
    )


class RunnerResult:
    def __init__(
        self,
        training_error_mean: float,
        training_error_std: float,
        test_error_mean: float,
        test_error_std: float,
        name: str,
        results: list[TrainerResult],
        test_error_histories: list[list[float]],
        test_error_histories_normalized: list[list[float]],
    ):
        self.training_error_mean = training_error_mean
        self.training_error_std = training_error_std
        self.test_error_mean = test_error_mean
        self.test_error_std = test_error_std
        self.name = name
        self.results = results
        self.test_error_histories = test_error_histories
        self.test_error_histories_normalized = test_error_histories_normalized


def run_n_times(
    n,
    get_datasets,
    config: TrainerConfig,
) -> RunnerResult:
    training_errors = []
    test_errors = []
    results = []
    test_error_histories = []
    test_error_histories_normalized = []

    for i in range(n):
        (
            train_dataset,
            train_dataset_outputs,
            test_dataset,
            test_dataset_outputs,
            name,
        ) = get_datasets()
        iterationResult = run_iteration(
            train_dataset,
            train_dataset_outputs,
            test_dataset,
            test_dataset_outputs,
            config,
        )

        training_errors.append(iterationResult.training_error)
        test_errors.append(iterationResult.test_error)
        results.append(iterationResult.result)
        test_error_histories.append(iterationResult.test_error_history)
        test_error_histories_normalized.append(iterationResult.test_error_history_normalized)
    return RunnerResult(
        training_error_mean=np.mean(training_errors),
        training_error_std=np.std(training_errors),
        test_error_mean=np.mean(test_errors),
        test_error_std=np.std(test_errors),
        name=name,
        results=results,
        test_error_histories=test_error_histories,
    )
