import random

from .individual import Individual
from colormath.color_objects import XYZColor
from .color_tools import similitude


# Selection functions don't work in place. They return the new final generation (list of Individual).
# Use-all implementation. Parents and children are considered in the selection.
# If the population limit has not yet been reached, parents and children are returned together. The wrapper function already does this.
# if population_limit > len(actual_gen) + len(new_gen): return next_gen

# __Fitness Function
# Returns fitness, relative fitness and accumulated fitness.
# TODO: Maybe we can store the fitness (after sum_fitness_reverse) in the Individual
def __fitness(gen: list[Individual], color_target: XYZColor):
    fitness = []
    sum_fitness = 0
    for i in range(len(gen)):
        fitness.append(similitude(gen[i].xyz, color_target))
        sum_fitness += fitness[i]

    sum_fitness_reverse = 0
    for i in range(len(gen)):
        fitness[i] = 1 / (fitness[i] / sum_fitness)
        sum_fitness_reverse += fitness[i]

    relative_fitness = []
    accumulated_fitness = []
    sum_accumulated_fitness = 0
    for i in range(len(fitness)):
        relative_fitness.append(fitness[i] / sum_fitness_reverse)
        sum_accumulated_fitness += relative_fitness[i]
        accumulated_fitness.append(sum_accumulated_fitness)

    return fitness, relative_fitness, accumulated_fitness


# Roulette Selection
# Uses r = random[0,1) as many times until (population_limit)-individuals have been selected.
# Choose the first individual i such that r < accumulated_fitness[i]
def roulette(actual_gen: list[Individual], new_gen: list[Individual], population_limit: int, color_target: XYZColor) -> \
        list[Individual]:
    next_gen = [j for i in [actual_gen, new_gen] for j in i]

    lr = []
    chosen = [False] * len(next_gen)
    randomly_chosen = 0
    fitness, relative_fitness, accumulated_fitness = __fitness(next_gen, color_target)

    while randomly_chosen < population_limit:
        r = random.random()
        for i in range(len(accumulated_fitness)):
            if r < accumulated_fitness[i] and not chosen[i]:
                chosen[i] = True
                randomly_chosen += 1
                lr.append(next_gen[i])
                break

    return lr


def elite(actual_gen: list[Individual], new_gen: list[Individual], population_limit: int, color_target: XYZColor) -> \
        list[Individual]:
    return 2


def det_tournaments(actual_gen: list[Individual], new_gen: list[Individual], population_limit: int,
                    color_target: XYZColor) -> list[Individual]:
    return 3


def prob_tournaments(actual_gen: list[Individual], new_gen: list[Individual], population_limit: int,
                     color_target: XYZColor) -> list[Individual]:
    return 4


selection_list = {
    "roulette": roulette,
    "elite": elite,
    "det_tournaments": det_tournaments,
    "prob_tournaments": prob_tournaments
}


def selection(config_selection: int, actual_gen: list[Individual], new_gen: list[Individual], population_limit: int,
              color_target: XYZColor) -> list[Individual]:
    if population_limit > len(actual_gen) + len(new_gen):
        return actual_gen + new_gen
    return selection_list[config_selection](actual_gen, new_gen, population_limit, color_target)
