import numpy as np
from colr import color
from typing import Callable
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Kohonen:
    def __init__(self, data: dict, k: int, initial_radius: float, initial_eta: float, max_iterations: int,
                 initialize_weights_method: Callable, print_neighborhood: bool, print_iteration_results: bool) -> None:
        self.winner_indexes = None
        self.grid = np.random.rand(k, k, len(next(iter(data.values()))))
        self.grid = self.grid.astype('float64')
        self.winners = np.zeros((k, k))
        self.winners_data = np.empty((k, k), dtype=np.ndarray)
        for i in range(k):
            for j in range(k):
                self.winners_data[i][j] = np.array([])
        self.data = data

        self.__initialize_weights_random(k, data)

        self.initial_radius = initial_radius
        self.radius = initial_radius
        self.initial_eta = initial_eta
        self.eta = initial_eta
        self.iteration = 0
        self.decay_factor = 0.9
        self.max_iterations = max_iterations
        self.print_neighborhood = print_neighborhood
        self.print_iteration_results = print_iteration_results

    # Primer metodo para inicializar los weights, random entre el minimo y el máximo de cada posición
    # contando todas las entradas
    def __initialize_weights_random(self, k: int, data: dict):
        # Para asegurarme que sean type float porque algunos quedan como object
        numeric_values = np.array(list(data.values()), dtype=float)

        # Obtener los valores mínimos y máximos de cada posición
        min_values = np.min(numeric_values, axis=0)
        max_values = np.max(numeric_values, axis=0)

        self.grid = np.clip(self.grid, min_values, max_values)

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

    # Empezar el algoritmo
    def start(self) -> dict[np.ndarray | int]:
        while self.iteration < self.max_iterations:
            self.iteration += 1

            random_key = np.random.choice(list(self.data.keys()))
            input_data = self.data[random_key]
            input_data = input_data.astype('float64')

            self.winner_indexes = self.__find_winner_neuron(input_data)
            self.winners[self.winner_indexes[0]][self.winner_indexes[1]] += 1

            #if not np.any(self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]] == random_key):
            self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]] = np.append(
                self.winners_data[self.winner_indexes[0]][self.winner_indexes[1]], random_key)

            self.__update_weights(input_data, self.winner_indexes)

            if self.print_iteration_results:
                self.__print_iteration_results()

            self.radius = max(self.initial_radius * (self.decay_factor ** self.iteration), 1)
            self.eta = max(self.initial_eta * (self.decay_factor ** self.iteration), 0.001)

        # self.__print_final_results()

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
        print(f"Iteration {self.iteration}")
        print(f"Radius: {self.radius}")
        print(f"Eta: {self.eta}")

        print(f"---- Winners Map ----")
        print(self.winners)

        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                print(self.winners_data[i][j], end='')
            print()
