import numpy as np
import matplotlib.pyplot as plt

from tp4.Oja.OjaPerceptron import OjaPerceptron
from tp4.Oja.OjaSimpleTrainer import OjaSimpleTrainer
from tp4.tools import csv_to_dict, standardize_data


class OjaRunner:
    def __init__(self, standardized_data):
        size = len(list(standardized_data.values())[0])
        initial_weights = np.random.uniform(-0.5, 0.5, size=size)
        self.perceptron = OjaPerceptron(initial_weights)
        self.trainer = OjaSimpleTrainer(self.perceptron, standardized_data)

    def run(self):
        self.trainer.train()
        return self.perceptron.w


countries, labels, country_data = csv_to_dict("../europe.csv")
standardized_country_data = standardize_data(country_data)
final_weights = OjaRunner(standardized_country_data).run()
data_labels = labels[1:]

pca1_values = []
for label, data_array in standardized_country_data.items():
    pca1 = np.dot(data_array, final_weights)
    pca1_values.append(pca1)
    print(f"PCA1 for {label}: {pca1}")

# Generate a bar plot
plt.figure(figsize=(10, 5))
plt.bar(standardized_country_data.keys(), pca1_values)
plt.xlabel('Countries')
plt.ylabel('PCA1')
plt.title('PCA1 values for each country')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.show()

