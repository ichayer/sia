from sklearn.decomposition import PCA
import numpy as np

from tp4.Oja.OjaPerceptron import OjaPerceptron
from tp4.Oja.OjaSimpleTrainer import OjaSimpleTrainer, Config
from tp4.Oja.pca_plots import plot_pca1
from tp4.tools import csv_to_dict, standardize_data


class OjaRunner:
    def __init__(self, standardized_data):
        size = len(list(standardized_data.values())[0])
        initial_weights = np.random.uniform(-0.5, 0.5, size=size)
        self.perceptron = OjaPerceptron(initial_weights)
        self.config = Config.load_from_file("config.json")
        self.trainer = OjaSimpleTrainer(self.perceptron, standardized_data, self.config)

    def run(self) -> OjaPerceptron:
        self.trainer.train()
        return self.perceptron


countries, labels, country_data = csv_to_dict("../europe.csv")
standardized_country_data = standardize_data(country_data)
perceptron = OjaRunner(standardized_country_data).run()
data_labels = labels[1:]

# SkLearn DATA
pca = PCA(n_components=1)
pca.fit_transform(np.array(list(standardized_country_data.values()), dtype=np.float64))
pca_sklearn_values = pca.components_[0]
# END SkLearn PCA

# Print weights
print(f"Final weights: {perceptron.w}")

# Average weights error
print(f"Average weights error: {np.average(np.abs(pca_sklearn_values - perceptron.w))}")

pca1_values = []
for label, data_array in standardized_country_data.items():
    pca1 = perceptron.evaluate(data_array)
    pca1_values.append(pca1)
    print(f"PC1 for {label}: {pca1}")

# Plot
plot_pca1(list(standardized_country_data.keys()), pca1_values)
