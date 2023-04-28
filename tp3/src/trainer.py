import json
from enum import Enum
import numpy as np
from .perceptron import Perceptron
from . import error_funcs


class TrainerConfig:
    """Encapsulates a configuration on how how to train a perceptron."""
    
    def __init__(self, error_func, acceptable_error, learning_rate=0.1, max_epochs=100, use_batch_increments=False, print_every=None, weight_comparison_epsilon=0.00001) -> None:
        self.error_func = error_func
        self.acceptable_error = acceptable_error
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_batch_increments = use_batch_increments
        self.print_every = print_every
        self.weight_comparison_epsilon = weight_comparison_epsilon
    
    def from_file(filename: str):
        with open(filename, "r") as f:
            config_dict = json.load(f)
        error_func = error_funcs.map[config_dict['error_func']]
        acceptable_error = float(config_dict['acceptable_error'])
        
        if acceptable_error < 0:
            raise Exception(f'Invalid configuration: acceptable_error must be greater than or equal to 0. Specified: {acceptable_error}')
        
        learning_rate = float(config_dict['learning_rate'])
        
        if learning_rate <= 0:
            raise Exception(f'Invalid configuration: learning_rate must be greater than 0. Specified: {learning_rate}')
        
        if 'max_epochs' in config_dict:
            max_epochs = int(config_dict['max_epochs'])
            if max_epochs < 0:
                raise Exception(f'Invalid configuration: max_epochs must be greater than 0. Specified: {max_epochs}')
        else:
            max_epochs = None
        
        if config_dict['weight_update_method'] == 'incremental':
            use_batch_increments = False
        elif config_dict['weight_update_method'] == 'batch':
            use_batch_increments = True
        else:
            raise Exception(f'Invalid configuration: weight_update_method must be either incremental or batch. Specified: {max_epochs}')
        
        print_every = int(config_dict['print_every']) if 'print_every' in config_dict else None
        weight_comparison_epsilon = float(config_dict['weight_comparison_epsilon']) if 'weight_comparison_epsilon' in config_dict else 0.00001

        return TrainerConfig(error_func, acceptable_error, learning_rate, max_epochs, use_batch_increments, print_every, weight_comparison_epsilon)


class EndReason(Enum):
    EPOCH_LIMIT_REACHED = "Epoch limit reached"
    ACCEPTABLE_ERROR_REACHED = "Acceptable error reached"
    WEIGHTS_HAVENT_CHANGED = "Weights haven't changed"


class TrainerResult:
    """Encapsulates the result of training a single perceptron."""
    
    def __init__(self, epoch_num: int, weights_history: list[np.ndarray[float]], error_history: list[float], end_reason: EndReason) -> None:
        self.epoch_num = epoch_num
        self.weights_history = weights_history
        self.error_history = error_history
        self.end_reason = end_reason


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
    end_reason = None

    while error > config.acceptable_error and (config.max_epochs is None or epoch_num < config.max_epochs) and end_reason is None:
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
        if np.abs(np.subtract(weights_history[-1], perceptron.w)).max() < config.weight_comparison_epsilon:
            end_reason = EndReason.WEIGHTS_HAVENT_CHANGED
        weights_history.append(np.copy(perceptron.w))
        
        error = evaluate_perceptron(perceptron, dataset, dataset_outputs, config.error_func, print_now)
        error_history.append(error)

    if end_reason is None:
        if error <= config.acceptable_error:
            end_reason = EndReason.ACCEPTABLE_ERROR_REACHED
        elif epoch_num == config.max_epochs:
            end_reason = EndReason.EPOCH_LIMIT_REACHED

    return TrainerResult(epoch_num, weights_history, error_history, end_reason)
