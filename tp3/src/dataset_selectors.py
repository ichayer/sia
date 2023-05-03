from numpy import random, arange


def split_random_by_percentage(dataset, dataset_outputs, percentage):
    dataset_size = len(dataset)
    indices = arange(dataset_size)
    random.shuffle(indices)
    split_index = int(dataset_size * percentage)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    train_dataset = [dataset[i] for i in train_indices]
    train_dataset_outputs = [dataset_outputs[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    test_dataset_outputs = [dataset_outputs[i] for i in test_indices]
    return train_dataset, train_dataset_outputs, test_dataset, test_dataset_outputs


"""
    This function is used to handpick a diverse training dataset.
    0: x1 = 0
    1: x2 = 0
    3: x3 = 0
    4: x1 maximo
    11: y maximo
    21: y minimo y x2 maximo
    17: x1 minimo
    14: x2 minimo
    12: x3 minimo
"""


def handpick_diverse_training_dataset(dataset, dataset_outputs):
    chosen_indexes = [0, 1, 3, 11, 21, 17, 14, 12]
    return handpick(dataset, dataset_outputs, chosen_indexes)


def handpick(dataset, dataset_outputs, chosen_idexes):
    train_dataset = [dataset[i] for i in chosen_idexes]
    train_dataset_outputs = [dataset_outputs[i] for i in chosen_idexes]
    test_dataset = [dataset[i] for i in range(len(dataset)) if i not in chosen_idexes]
    test_dataset_outputs = [
        dataset_outputs[i]
        for i in range(len(dataset_outputs))
        if i not in chosen_idexes
    ]
    return train_dataset, train_dataset_outputs, test_dataset, test_dataset_outputs
