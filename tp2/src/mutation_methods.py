import math
import random
from .individual import Individual


# Mutations methods don't return anything. They work in place with the population variable.

def gen(population: list[Individual], generation: int) -> None:
    return 1


def limited_multigen(population: list[Individual], generation: int) -> None:
    return 2


def uniform_multigen(population: list[Individual], generation: int) -> None:
    return 3


# Complete Mutation
# A random sign is chosen (to subtract or add). If it is possible to carry out that operation, it is done, otherwise, none is done.
# Up to half of the genes are chosen at random (inside while). Then it goes through all the genes not chosen inside the while (inside for)

def complete(population: list[Individual], generation: int) -> None:
    r = random.uniform(0, 100)
    probability = max(100 / math.pow(2, generation),
                      2)  # TODO: bring these variables (prob and change) to the config (and remove generation parameter if possible)
    # If generation > 1024, pow(2,1024) doesn't work.
    change = max(100 / math.pow(2, generation), 5)
    if probability >= r:
        for individual in population:
            sum_ratios = 0
            randomly_chosen = 0
            chosen = [False] * len(individual.color_ratios)

            # Up to half of the genes are chosen at random
            while randomly_chosen <= len(individual.color_ratios) / 2:
                r = random.randint(0, len(individual.color_ratios) - 1)
                if not chosen[r]:
                    randomly_chosen += 1
                    s = random.choice([1, -1])
                    if 0 < individual.color_ratios[r][1] + change * s < 100:
                        individual.color_ratios[r][1] += change * s
                    sum_ratios += individual.color_ratios[r][1]
                    chosen[r] = True

            # Then it goes through all the genes not chosen inside the previous while
            for i in range(len(individual.color_ratios)):
                if not chosen[i]:
                    s = random.choice([1, -1])
                    if 0 < individual.color_ratios[i][1] + change * s < 100:
                        individual.color_ratios[i][1] += change * s
                    sum_ratios += individual.color_ratios[r][1]

            individual.normalize()


mutation_list = {
    "gen": gen,
    "limited_multigen": limited_multigen,
    "uniform_multigen": uniform_multigen,
    "complete": complete
}


def mutation(config_mutation: int, population: list[Individual], generation: int) -> None:
    return mutation_list[config_mutation](population, generation)
