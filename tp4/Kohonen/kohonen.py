import numpy as np
from colr import color
from typing import Callable
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


class Kohonen:
    def __init__(self, data: dict, k: int, initial_radius: float, initial_eta: float, max_iterations: int,
                 initialize_weights_method: Callable, print_neighborhood: bool, print_iteration_results: bool,
                 print_final_results: bool, threshold: float, decay_factor: float) -> None:
        self.winner_indexes = None
        self.grid = np.random.rand(k, k, len(next(iter(data.values()))))
        self.grid = self.grid.astype('float64')
        self.winners = np.zeros((k, k))
        self.winners_data = np.empty((k, k), dtype=np.ndarray)
        for i in range(k):
            for j in range(k):
                self.winners_data[i][j] = np.array([])
        self.data = data

        self.__initialize_weights_random()

        self.k = k
        self.initial_radius = initial_radius
        self.radius = initial_radius
        self.initial_eta = initial_eta
        self.eta = initial_eta
        self.decay_factor = decay_factor
        self.iteration = 0
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.print_neighborhood = print_neighborhood
        self.print_iteration_results = print_iteration_results
        self.print_final_results = print_final_results

    # Primer metodo para inicializar los weights, random entre el minimo y el máximo de cada posición
    # contando todas las entradas
    def __initialize_weights_random(self):
        # Para asegurarme que sean type float porque algunos quedan como object
        numeric_values = np.array(list(self.data.values()), dtype=float)

        # Obtener los valores mínimos y máximos de cada posición
        min_values = np.min(numeric_values, axis=0)
        max_values = np.max(numeric_values, axis=0)

        for i in range(self.k):
            for j in range(self.k):
                self.grid[i][j] = np.random.uniform(min_values, max_values)

    # Recibe el input random elegido y encuentra la neurona que tiene los weights más parecidos
    def __find_winner_neuron(self, input_data: np.ndarray) -> tuple[int, int]:
        distance = None
        winner_indexes = [0, 0]
        for (i, row) in enumerate(self.grid):
            for (j, neuron) in enumerate(row):
                d = np.linalg.norm(neuron - input_data)
                if not distance or distance > d:
                    distance = d
                    winner_indexes = [i, j]

        return tuple(winner_indexes)

    def __update_weights(self, input_data: np.ndarray, winner_indexes: tuple) -> None:
        self.previous_grid = np.copy(self.grid)
        rows, cols, weights = self.grid.shape
        neighborhood = np.zeros((rows, cols))
        neighborhood[winner_indexes[0]][winner_indexes[1]] = 2
        line = "-----" * cols

        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - winner_indexes[0]) ** 2 + (j - winner_indexes[1]) ** 2)
                if distance != 0 and distance <= self.radius:
                    self.grid[i][j] += self.eta * (input_data - self.grid[i][j])
                    neighborhood[i][j] = 1

        if self.print_neighborhood:
            print("--- Neighborhood ---")
            print(f"{line}")
            for i in range(rows):
                for j in range(cols):
                    if j == 0:
                        print("|", end='')
                    if neighborhood[i][j] == 1:
                        print(f"{color('  ', fore=(0, 0, 0), back=(255, 255, 255))} | ", end='')
                    if neighborhood[i][j] == 0:
                        print(f"{color('  ', fore=(0, 0, 0), back=(0, 0, 0))} | ", end='')
                    if neighborhood[i][j] == 2:
                        print(f"{color('  ', fore=(0, 0, 0), back=(0, 0, 255))} | ", end='')
                print(f"\n{line}")

    def __has_converged(self) -> bool:
        if self.iteration == 0:
            return False

        return np.allclose(self.grid, self.previous_grid, atol=self.threshold)

    # Empezar el algoritmo
    def start(self) -> dict[np.ndarray | int]:
        while self.iteration < self.max_iterations and not self.__has_converged():
            self.iteration += 1

            random_key = np.random.choice(list(self.data.keys()))
            input_data = self.data[random_key]
            input_data = input_data.astype('float64')

            self.winner_indexes = self.__find_winner_neuron(input_data)
            self.winners[self.winner_indexes[0]][self.winner_indexes[1]] += 1

            # if not np.any(self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]] == random_key):
            self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]] = np.append(
                self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]], random_key)

            self.__update_weights(input_data, self.winner_indexes)

            if self.print_iteration_results:
                self.__print_iteration_results()

            self.radius = max(self.initial_radius * (self.decay_factor ** self.iteration), 1)
            self.eta = max(self.initial_eta * (self.decay_factor ** self.iteration), 0.001)
            # self.eta = self.initial_eta/self.iteration

        if self.print_final_results:
            self.__print_final_results()

        return {"matrix_winners": self.winners, "matrix_winners_data": self.winners_data, "iteration": self.iteration}

    def __print_iteration_results(self):
        print(f"Iteration {self.iteration}")
        print(f"Radius: {self.radius}")
        print(f"Eta: {self.eta}")

        print("----- Weight Grid -----")
        print(self.grid)

        print(f"Winner Neuron: {self.winner_indexes}")
        print(f"{self.grid[self.winner_indexes[0]][self.winner_indexes[1]]}")

    def __print_final_results(self):
        print("---- Final Results ----")
        print(f"Iteration {self.iteration}")
        print(f"Radius: {self.radius}")
        print(f"Eta: {self.eta}")

        print("---- Grid ----")
        print(self.grid)

        print(f"---- Winners Map ----")
        print(self.winners)

    def plot_heatmap_u_matrix(self) -> None:
        # Obtén las dimensiones del mapa
        n, m, d = self.grid.shape

        # Crea una matriz "u" de tamaño "n x m" inicializada con ceros
        u = np.zeros((n, m))

        # Itera sobre cada neurona en el mapa
        for i in range(n):
            for j in range(m):
                # Calcula los índices de las neuronas vecinas
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

                # Calcula la suma de las distancias euclidianas a las neuronas vecinas
                dist_sum = 0.0
                num_neighbors = 0

                for neighbor in neighbors:
                    ni, nj = neighbor
                    if 0 <= ni < n and 0 <= nj < m:
                        # Calcula la distancia euclidiana entre los vectores de pesos
                        dist = np.linalg.norm(self.grid[i, j] - self.grid[ni, nj])
                        dist_sum += dist
                        num_neighbors += 1

                # Calcula el promedio de las distancias euclidianas
                if num_neighbors > 0:
                    u[i, j] = dist_sum / num_neighbors

        # La matriz "u" contiene el promedio de las distancias euclidianas entre
        # el vector de pesos de cada neurona y los vectores de pesos de las neuronas vecinas.

        fig, ax = plt.subplots(figsize=(10, 10))
        heatmap = ax.imshow(u, cmap="gray_r")
        cbar = ax.figure.colorbar(heatmap, ax=ax, shrink=0.5)
        plt.title('Average error between neighbours', fontsize=24)

        # Mostrar el heatmap
        plt.show()

    def plot_heatmap_final_entries(self, delete_smalls: bool) -> None:
        countries_no_dup = np.empty((self.k, self.k), dtype=np.ndarray)

        for i in range(self.k):
            for j in range(self.k):
                countries_no_dup[i][j] = np.unique(self.winners_data[i][j])

        # Crear el heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        heatmap = ax.imshow(self.winners, cmap="YlGn")
        cbar = ax.figure.colorbar(heatmap, ax=ax, shrink=0.5).set_label(label='Entries Amount', size=20)
        plt.title('Final entries per neuron', fontsize=24)

        for i in range(self.k):
            for j in range(self.k):
                str_annotate = ""
                dict_countries = {}
                for country in countries_no_dup[i, j]:
                    quantity = np.count_nonzero(self.winners_data[i, j] == country)
                    dict_countries[country] = quantity

                if len(dict_countries) > 0 and delete_smalls:
                    max_value = max(dict_countries.values())
                    threshold = 0.1 * max_value
                    keys_to_eliminate = [key for key, valor in dict_countries.items() if valor < threshold]
                    for key in keys_to_eliminate:
                        del dict_countries[key]

                for key, value in dict_countries.items():
                    str_annotate += f'{key[:3]}: {value}\n'

                value = self.winners[i, j]
                color_bg = 'white' if value > self.max_iterations / 10 else 'black'
                plt.annotate(str_annotate.rstrip("\n"), (j, i), color=color_bg, ha='center', va='center')

        # Mostrar el heatmap
        plt.show()

    def plot_heatmap_variable(self, variable_index: int, variable_name: str):
        if variable_index < 0 or variable_index > len(next(iter(self.data.values()))):
            raise TypeError("variable_index out of index")

        extracted_variables = self.grid[:, :, variable_index]

        fig, ax = plt.subplots(figsize=(10, 10))
        heatmap = ax.imshow(extracted_variables, cmap="PuBu")
        cbar = ax.figure.colorbar(heatmap, ax=ax).set_label(label='Entries Amount', size=20)
        plt.title(f'Variable {variable_name}', fontsize=24)

        plt.show()
