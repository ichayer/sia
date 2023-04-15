import math
import random
from tp2.src.individual import Individual


def gen(population: list[Individual], generation: int) -> None:
    return 1


def limited_multigen(population: list[Individual], generation: int) -> None:
    return 2


def uniform_multigen(population: list[Individual], generation: int) -> None:
    return 3


def complete(population: list[Individual], generation: int) -> None:
    r = random.uniform(0, 100)
    probability = max(100 / math.pow(2, generation), 2)
    change = max(100 / math.pow(2, generation), 5)
    if probability >= r:
        for individual in population:
            sum_ratios = 0
            randomly_chosen = 0
            chosen = [False] * len(individual.color_ratios)

            while randomly_chosen <= len(individual.color_ratios) - 1:
                r = random.randint(0, len(individual.color_ratios) - 1)
                if not chosen[r]:
                    randomly_chosen += 1
                    s = random.choice([1, -1])
                    if 0 < individual.color_ratios[r][1] + change * s < 100:
                        individual.color_ratios[r][1] += change * s
                    sum_ratios += individual.color_ratios[r][1]
                    chosen[r] = True

            for i in range(len(individual.color_ratios)):
                if not chosen[i]:
                    s = random.choice([1, -1])
                    if 0 < individual.color_ratios[i][1] + change * s < 100:
                        individual.color_ratios[i][1] += change * s
                    sum_ratios += individual.color_ratios[r][1]

            individual.normalize()


mutation_list = [gen, limited_multigen, uniform_multigen, complete]


def mutation(config_mutation: int, population: list[Individual], generation: int) -> None:
    return mutation_list[config_mutation](population, generation)
