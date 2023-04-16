import random
import time
from colormath.color_conversions import convert_color
from colormath.color_objects import XYZColor, AdobeRGBColor
from tp2.src.individual import Individual
from tp2.src.crossover_methods import crossover
from tp2.src.mutation_methods import mutation
from tp2.src.selection_methods import selection
from tp2.src.finish_methods import finish
from tp2.src.color_tools import similitude


# Parameters are received as RGB for simplicity and user convenience, but we save them as XYZ
# It is in charge of saving all the parameters and iterating in the generations up to the cutting criterion.
# Once an instance is created, the start function must be called
class ColorGeneticAlgorithm:
    generation = int
    actual_gen = []

    def __init__(self, color_set: list[AdobeRGBColor], color_target: AdobeRGBColor, config: dict) -> None:
        self.color_set = list(
            map(lambda x: convert_color(x, XYZColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50'),
                color_set))
        self.color_target = convert_color(color_target, XYZColor, through_rgb_type=AdobeRGBColor,
                                          target_illuminant='d50')
        self.population = config["population"] if "population" in config else 15
        self.selection_method = config["selection_method"] if "selection_method" in config else 0
        self.crossover_method = config["crossover_method"] if "crossover_method" in config else 0
        self.mutation_method = config["mutation_method"] if "mutation_method" in config else 3
        self.finish_method = config["finish_method"] if "finish_method" in config else [0]
        self.finish_parameters = config["finish_parameters"] if "finish_parameters" in config else {
            "generation": 15,
            "seconds": 5,
            "distance": 0.4
        }

        print(self.population)

    def __gen0(self) -> None:
        ratios = []
        self.generation = 0

        for color_xyz in self.color_set:
            ratios.append([color_xyz, 0])

        ratios[0][1] = 100
        self.actual_gen.append(Individual(ratios))

        for i in range(1, len(self.color_set)):
            ratios[i - 1][1] = 0
            ratios[i][1] = 100
            self.actual_gen.append(Individual(ratios))

        print("Generation 0:\n")
        for i, ind in enumerate(self.actual_gen):
            print(f"Ind {i}: \t {ind}")

    def start(self) -> None:
        start_time = time.time()
        # Generation 0
        self.__gen0()

        print("\nStarting the algorithm...\n")

        distance = [self.finish_parameters["distance"] + 1]
        while not finish(self.finish_method, self.finish_parameters, start_time, self.generation, min(distance)):
            # Generation i
            self.generation += 1
            new_gen = []

            print(f"\nGeneration {self.generation}:")

            # Crossover
            random.shuffle(self.actual_gen)
            for i, ind in enumerate(self.actual_gen):
                l = crossover(self.selection_method,
                              [self.actual_gen[i], self.actual_gen[(i + 1) % len(self.actual_gen)]])
                for j in range(len(l)):
                    new_gen.append(l[j])

            """ # Shows individual after crossover 
            print("\nCrossover:\n")
            for i, ind in enumerate(new_gen):
                print(f"Ind{i}: \t {ind}")
            """

            # Mutation
            mutation(self.mutation_method, new_gen, self.generation)

            """ # Shows individual after mutation
            print("\nMutation:\n")
            for i, ind in enumerate(new_gen):
                print(f"Ind{i}: \t {ind}")
            """

            # Selection
            self.actual_gen = selection(self.selection_method, self.actual_gen, new_gen, self.population,
                                        self.color_target)
            """
            print("\nSelection:\n")
            """
            distance = []
            for i, ind in enumerate(self.actual_gen):
                distance.append(similitude(ind.xyz, self.color_target))
                print(f"IND{i}: \t {ind} DIST:{str(round(distance[i], 2))}")
