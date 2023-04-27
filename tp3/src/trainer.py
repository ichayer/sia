import numpy as np
from .perceptron import Perceptron


class TrainerConfig:
    """Encapsulates a configuration on how how to train a perceptron."""
    
    def __init__(self, error_func, acceptable_error, learning_rate=0.1, max_epochs=100, use_batch_increments=False, print_every=None) -> None:
        self.error_func = error_func
        self.acceptable_error = acceptable_error
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_batch_increments = use_batch_increments
        self.print_every = print_every


class TrainerResult:
    """Encapsulates the result of training a single perceptron."""
    
    def __init__(self, epoch_num: int, weights_history: list[np.ndarray[float]], error_history: list[float]) -> None:
        self.epoch_num = epoch_num
        self.weights_history = weights_history
        self.error_history = error_history


def evaluate_perceptron(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float], error_func, print_output: bool) -> int:
    """
    Evaluates a perceptron with a given dataset.
    Returns: The amount of inputs in the dataset for which the perceptron returned the correct result.
    """
    
    outputs = np.zeros(len(dataset))
    for i in range(len(dataset)):
        output = perceptron.evaluate(dataset[i])
        expected = dataset_outputs[i]
        outputs[i] = output
        if print_output:
            print(f"[{i}] {'✅' if output == expected else '❌'} expected: {expected} got: {output} data: {dataset[i]}")
    return error_func(dataset_outputs, outputs)


def train_perceptron(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float], config: TrainerConfig) -> TrainerResult:
    dataset_with_ones = [np.concatenate(([1], d)) for d in dataset]
    
    error = evaluate_perceptron(perceptron, dataset, dataset_outputs, config.error_func, False)
    
    epoch_num = 0
    weights_history = [np.copy(perceptron.w)]
    error_history = [error]

    while error > config.acceptable_error and epoch_num < config.max_epochs:
        epoch_num += 1
        
        for i in range(len(dataset)):
            perceptron.evaluate_and_adjust(dataset_with_ones[i], dataset_outputs[i], config.learning_rate)
            if not config.use_batch_increments:
                perceptron.update_weights()

        print_now = False
        if config.print_every is not None and epoch_num % config.print_every == 0:
            print("--------------------------------------------------")
            print(f"RESULTS AFER EPOCH {epoch_num} (weights {perceptron.w})")
            print_now = True
        
        if config.use_batch_increments:
            perceptron.update_weights()
        weights_history.append(np.copy(perceptron.w))
        
        error = evaluate_perceptron(perceptron, dataset, dataset_outputs, config.error_func, print_now)
        error_history.append(error)

    return TrainerResult(epoch_num, weights_history, error_history)
