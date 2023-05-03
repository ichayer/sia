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


