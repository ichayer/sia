import json
from enum import Enum
from typing import Optional, Dict, List
import numpy as np
from numpy import ndarray
import time

from .perceptron import Perceptron, MultilayerPerceptron
from . import error_funcs, theta_funcs
from .scaler import Scaler


class TrainerConfig:
    """Encapsulates a configuration on how to train a perceptron."""

    def __init__(self, theta: theta_funcs.ThetaFunction, error_func, acceptable_error, scaler: Scaler = Scaler(),
                 learning_rate=0.1, max_epochs=100, use_batch_increments=False, print_every=None,
                 weight_comparison_epsilon=0.00001) -> None:
        self.theta = theta
        self.scaler = scaler
        self.error_func = error_func
        self.acceptable_error = acceptable_error
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_batch_increments = use_batch_increments
        self.print_every = print_every
        self.weight_comparison_epsilon = weight_comparison_epsilon

    def from_dict(config_dict: dict):
        theta_config = config_dict['theta_config'] if 'theta_config' in config_dict else None
        theta = theta_funcs.map[config_dict['theta']](theta_config)

        scaler = Scaler.from_dict(config_dict['scaler'], theta) if 'scaler' in config_dict else Scaler()

        error_func = error_funcs.map[config_dict['error_func']]
        acceptable_error = float(config_dict['acceptable_error'])

        if acceptable_error < 0:
            raise Exception(
                f'Invalid configuration: acceptable_error must be greater than or equal to 0. Specified: {acceptable_error}')

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
            raise Exception(
                f'Invalid configuration: weight_update_method must be either incremental or batch. Specified: {max_epochs}')

        print_every = int(config_dict['print_every']) if 'print_every' in config_dict else None
        weight_comparison_epsilon = float(
            config_dict['weight_comparison_epsilon']) if 'weight_comparison_epsilon' in config_dict else 0.00001

        return TrainerConfig(theta, error_func, acceptable_error, scaler, learning_rate, max_epochs,
                             use_batch_increments, print_every, weight_comparison_epsilon)

    def from_file(filename: str):
        with open(filename, "r") as f:
            config_dict = json.load(f)
        return TrainerConfig.from_dict(config_dict)


class EndReason(Enum):
    EPOCH_LIMIT_REACHED = "Epoch limit reached"
    ACCEPTABLE_ERROR_REACHED = "Acceptable error reached"
    WEIGHTS_HAVENT_CHANGED = "Weights haven't changed"


class TrainerResult:
    """Encapsulates the result of training a single perceptron."""

    def __init__(self, epoch_num: int, weights_history: list[np.ndarray[float]], error_history: list[float],
                 end_reason: EndReason, test_error_history: Optional[list[float]] = None) -> None:
        self.epoch_num = epoch_num
        self.weights_history = weights_history
        self.error_history = error_history
        self.end_reason = end_reason
        self.test_error_history = test_error_history


class MultilayerTrainerResult:

    def __init__(self, epoch_num: int, weights_history, error_history, end_reason) -> None:
        self.epoch_num = epoch_num
        self.weights_history = weights_history
        self.error_history = error_history
        self.end_reason = end_reason


def evaluate_perceptron(perceptron: Perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float],
                        error_func, scaler: Scaler, print_output: bool, acceptable_error=0) -> int:
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
            err = error_func(scaler.reverse(np.array([expected])), scaler.reverse(np.array([output])))
            print(
                f"[{i}] {'✅' if err <= acceptable_error else '❌'} expected: {expected} got: {output} data: {dataset[i]}")
    return error_func(scaler.reverse(dataset_outputs), scaler.reverse(outputs))


def train_perceptron(perceptron: Perceptron,
                     dataset: list[np.ndarray[float]],
                     dataset_outputs: list[float],
                     config: TrainerConfig,
                     test_dataset: Optional[list[np.ndarray[float]]] = None,
                     test_dataset_outputs: Optional[list[float]] = None) -> TrainerResult:
    dataset_with_ones = [np.concatenate(([1], d)) for d in dataset]

    error = evaluate_perceptron(perceptron, dataset, dataset_outputs, config.error_func, config.scaler, False,
                                config.acceptable_error)
    test_error = evaluate_perceptron(perceptron, test_dataset, test_dataset_outputs, config.error_func, config.scaler,
                                     False, config.acceptable_error) if test_dataset is not None else None

    epoch_num = 0
    weights_history = [np.copy(perceptron.w)]
    error_history = [error]
    test_error_history = [test_error] if test_error is not None else None
    end_reason = None

    while error > config.acceptable_error and (
            config.max_epochs is None or epoch_num < config.max_epochs) and end_reason is None:
        epoch_num += 1

        for i in range(len(dataset)):
            perceptron.evaluate_and_adjust(dataset_with_ones[i], dataset_outputs[i], config.learning_rate)
            if not config.use_batch_increments:
                # TODO: Results history
                perceptron.update_weights()

        if config.use_batch_increments:
            perceptron.update_weights()

        print_now = False
        if config.print_every is not None and epoch_num % config.print_every == 0:
            print("--------------------------------------------------")
            print(f"RESULTS AFTER EPOCH {epoch_num} (weights {perceptron.w})")
            print_now = True

        if np.abs(np.subtract(weights_history[-1], perceptron.w)).max() < config.weight_comparison_epsilon:
            end_reason = EndReason.WEIGHTS_HAVENT_CHANGED
        weights_history.append(np.copy(perceptron.w))

        error = evaluate_perceptron(perceptron, dataset, dataset_outputs, config.error_func, config.scaler, print_now,
                                    config.acceptable_error)

        if test_dataset is not None:
            test_error = evaluate_perceptron(perceptron, test_dataset, test_dataset_outputs, config.error_func,
                                             config.scaler, print_now, config.acceptable_error)
            test_error_history.append(test_error)

        if error > error_history[-1]:
            print(f"⚠⚠⚠ WARNING! Error from epoch {epoch_num} has increased relative to previous epoch!")
        error_history.append(error)

    if end_reason is None:
        if error <= config.acceptable_error:
            end_reason = EndReason.ACCEPTABLE_ERROR_REACHED
        elif epoch_num == config.max_epochs:
            end_reason = EndReason.EPOCH_LIMIT_REACHED

    return TrainerResult(epoch_num, weights_history, error_history, end_reason, test_error_history)


def evaluate_multilayer_perceptron(multilayer_perceptron: MultilayerPerceptron, dataset: list[list[int]],
                                   dataset_outputs: list[list[int]],
                                   print_output: bool) -> dict[str, float | ndarray | list[list[int]]]:
    """
    Evaluates a multilayer perceptron with a given dataset.
    Returns: The amount of inputs in the dataset for which the perceptron returned the correct result.
    """
    err = 0
    last_layer_results = []
    for (i, data) in enumerate(dataset):
        last_layer_result = np.array(multilayer_perceptron.feed_forward(data))
        last_layer_results.append(last_layer_result)
        expected = np.array(dataset_outputs[i])
        err += np.power(expected - last_layer_result, 2)
        if print_output:
            print(f"[Data {i + 1}] expected: {dataset_outputs[i]} got: {last_layer_result} data: {dataset[i]}")
    return {"err": err * 0.5, "expected_output": dataset_outputs, "network_output": last_layer_results}


def train_multilayer_perceptron(multilayer_perceptron: MultilayerPerceptron, dataset: list[np.ndarray[float]],
                                config: TrainerConfig) -> MultilayerTrainerResult:
    epoch_num = 0
    error_history = []
    weights_history = []
    result_history = []

    weights_history.append([])
    for (i, perceptron) in enumerate(multilayer_perceptron.last_layer):
        weights_history[-1].append(None)
        weights_history[-1][i] = np.copy(perceptron.w)

    end_reason = None
    average_epoch_elapsed_time = 0

    while (config.max_epochs is None or epoch_num < config.max_epochs) and end_reason is None:
        start_time = time.time()
        epoch_num += 1
        error = 0

        result_history.append([])
        weights_history.append([])
        for i in range(len(dataset)):
            multilayer_perceptron.evaluate_and_adjust(dataset[i], config.learning_rate)
            result_history[-1].append([])
            for (j, perceptron) in enumerate(multilayer_perceptron.last_layer):
                result_history[-1][-1].append(None)
                result_history[-1][i][j] = multilayer_perceptron.results[-1][j]
                error += np.power(dataset[i][j] - result_history[epoch_num - 1][i][j], 2)
            if not config.use_batch_increments:
                multilayer_perceptron.update_weights()

        if config.use_batch_increments:
            multilayer_perceptron.update_weights()

        error = error * 0.5

        should_print = config.print_every is not None and epoch_num % config.print_every == 0
        # Print weights
        if should_print:
            print(f"Epoch {epoch_num}")
            print(f"Average elapsed time {average_epoch_elapsed_time * 1000}ms")
            print(f"Error: {error}")
            # print_epoch(dataset, epoch_num, multilayer_perceptron, result_history)

        weights_not_changed = True
        for (i, perceptron) in enumerate(multilayer_perceptron.last_layer):
            weights_history[-1].append(None)
            weights_history[-1][i] = np.copy(perceptron.w)
            if np.abs(np.subtract(weights_history[-2][i], perceptron.w)).max() >= config.weight_comparison_epsilon:
                weights_not_changed = False
        if weights_not_changed:
            end_reason = EndReason.WEIGHTS_HAVENT_CHANGED

        if should_print and len(error_history) != 0 and error > error_history[-1]:
            print(f"⚠⚠⚠ WARNING! Error from epoch {epoch_num} has increased relative to previous epoch!")

        error_history.append(error)

        if end_reason is None:
            if error <= config.acceptable_error:
                end_reason = EndReason.ACCEPTABLE_ERROR_REACHED
            elif epoch_num == config.max_epochs:
                end_reason = EndReason.EPOCH_LIMIT_REACHED
        end_time = time.time()
        elapsed_time = end_time - start_time
        average_epoch_elapsed_time = (average_epoch_elapsed_time + elapsed_time) / 2

    return MultilayerTrainerResult(epoch_num, weights_history, error_history, end_reason)


def print_epoch(dataset, epoch_num, multilayer_perceptron, result_history):
    print("--------------------------------------------------")
    for perceptron in multilayer_perceptron.last_layer:
        print(f"RESULTS AFTER EPOCH {epoch_num} (weights {perceptron.w})")
    # Print results
    for i in range(len(dataset)):
        for (j, perceptron) in enumerate(multilayer_perceptron.last_layer):
            print(
                f"[Data {i}, Neuron Output {j}] expected: {dataset[i][j]} got: {result_history[epoch_num - 1][i][j]} data: {dataset[i]}")
