from tp4.tools import *
from tp4.Kohonen.kohonen import *
import json
import matplotlib.pyplot as plt
import seaborn as sns

country_data = csv_to_dict("europe.csv")
standardized_country_data = standardize_data(country_data)

with open('kohonen-config.json') as f:
    config = json.load(f)

# Acceder a los valores del archivo JSON
k = config['k']
initial_radius = config['initial_radius']
initial_eta = config['initial_eta']
print_neighborhood = config['print_neighborhood']
print_iteration_results = config['print_iteration_results']
initialize_weights_method = config['initialize_weights_method']
max_iterations = config['max_iterations']


kohonen = Kohonen(standardized_country_data, k, initial_radius, initial_eta, max_iterations, initialize_weights_method, print_neighborhood, print_iteration_results, )
result = kohonen.start()

print("Results at Iteration: ", end='')
print(result["iteration"])

winners = result["matrix_winners"]
countries = result["matrix_winners_data"]
countries_no_dup = np.empty((k, k), dtype=np.ndarray)

for i in range(k):
    for j in range(k):
        countries_no_dup[i][j] = np.unique(countries[i][j])

# Crear el heatmap
fig = plt.figure(figsize=(10, 10))
heatmap = plt.imshow(winners, cmap='hot', interpolation='nearest')

print(countries_no_dup)


for i in range(k):
    for j in range(k):
        str_annotate = ""
        for country in countries_no_dup[i, j]:
            quantity = np.count_nonzero(countries[i, j] == country)
            str_annotate += f'{country[:3]}: {quantity}\n'
        value = winners[i, j]
        color = 'white' if value < 10 else 'black'
        plt.annotate(str_annotate.rstrip("\n"), (j, i), color=color, ha='center', va='center')

plt.title('Final entries per neuron', fontsize=24)

plt.colorbar(heatmap).set_label(label='Entries Amount', size=20)

# Mostrar el heatmap
plt.show()