import numpy as np
from .perceptron import perceptron


def evaluate_perceptron(perceptron: perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float], print_output: bool) -> int:
    """
    Evaluates a perceptron with a given dataset.
    Returns: The amount of inputs in the dataset for which the perceptron returned the correct result.
    """
    
    amount_ok = 0
    for i in range(len(dataset)):
        output = perceptron.evaluate(dataset[i])
        expected = dataset_outputs[i]
        amount_ok += 1 if expected == output else 0
        if print_output:
            print(f"[{i}] {'✅' if output == expected else '❌'} expected: {expected} got: {output} data: {dataset[i]}")
    return amount_ok


def train_perceptron(perceptron: perceptron, dataset: list[np.ndarray[float]], dataset_outputs: list[float], learning_rate: 0.1, max_epochs=100, use_batch_increments=False, print_every=None):
    dataset_with_ones = [np.concatenate(([1], d)) for d in dataset]

    for epoch_idx in range(1, max_epochs+1):
        for i in range(len(dataset)):
            perceptron.evaluate_and_adjust(dataset_with_ones[i], dataset_outputs[i], learning_rate)
            if not use_batch_increments:
                perceptron.update_weights()

        if print_every is not None and epoch_idx % print_every == 0:
            print("--------------------------------------------------")
            print(f"RESULTS AFER EPOCH {epoch_idx} (weights {perceptron.w})")
        amount_ok = evaluate_perceptron(perceptron, dataset, dataset_outputs, epoch_idx % print_every == 0)
        if amount_ok == len(dataset):
            break
        
        if use_batch_increments:
            perceptron.update_weights()

    if print_every is not None and epoch_idx < print_every:
        print("...")
