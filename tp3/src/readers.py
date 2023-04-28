import numpy as np
import csv

def read_csv(filename: str) -> tuple[list[np.ndarray[float]], list[float]]:
    dataset = []
    dataset_outputs = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if reader.line_num != 1:
                dataset.append(np.array(row[:-1]).astype(float))
                dataset_outputs.append(float(row[-1]))

    return dataset, dataset_outputs
