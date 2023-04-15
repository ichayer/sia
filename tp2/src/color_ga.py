from colormath.color_conversions import convert_color
from colormath.color_objects import XYZColor, AdobeRGBColor
from tp2.src.color_tools import MyXYZColor
from tp2.src.individual import Individual
from tp2.src.crossover_methods import crossover
from tp2.src.mutation_methods import mutation
from tp2.src.selection_methods import selection


# parameters are received as RGB for simplicity, but we saved as XYZ
class ColorGeneticAlgorithm:
    generation = int
    actual_gen = []

    def __init__(self, color_set: list[AdobeRGBColor], color_target: AdobeRGBColor, selection_method: int,
                 crossover_method: int,
                 mutation_method: int, finish: int, population: int) -> None:
        self.color_set = list(
            map(lambda x: convert_color(x, XYZColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50'),
                color_set))
        self.color_target = convert_color(color_target, XYZColor, through_rgb_type=AdobeRGBColor,
                                          target_illuminant='d50')
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.finish_method = finish
        self.population = population

    def __gen0(self) -> None:
        ratios = []
        self.generation = 0

        for color_xyz in self.color_set:
            ratios.append([MyXYZColor(color_xyz), 0])

        ratios[0][1] = 100
        self.actual_gen.append(Individual(ratios))

        for i in range(1, len(self.color_set)):
            ratios[i - 1][1] = 0
            ratios[i][1] = 100
            self.actual_gen.append(Individual(ratios))

        for i, ind in enumerate(self.actual_gen):
            print(f"Individual {i}: {ind}")

    def start(self) -> None:
        print("Starting the algorithm...")

        # Generation 0
        self.__gen0()

        # Generation i
        self.generation += 1
        new_gen = []

        # Crossover
        for i, ind in enumerate(self.actual_gen):
            l = crossover(self.selection_method, [self.actual_gen[i], self.actual_gen[(i + 1) % len(self.actual_gen)]])
            new_gen.append(l[0])
            new_gen.append(l[1])

        for i, ind in enumerate(new_gen):
            print(f"Ind{i}: \t {new_gen[i]}")

        # Mutation
        mutation(self.mutation_method, new_gen, self.generation)

        for i, ind in enumerate(new_gen):
            print(f"Ind{i}: \t {new_gen[i]}")
