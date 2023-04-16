import math
import random
from .individual import Individual
from typing import List, Tuple


# Crossover functions return one or two children. Only one child if one of them has all genes at 0% ("Dead individual")
# To do this, the ratios of the colors are added in a variable as they are iterated.
def _crossover(parent1: Individual, parent2: Individual, crossover_points: List[Tuple[int, int]]) -> List[Individual]:
    children = []
    ratios1 = []
    ratios2 = []
    sum1 = 0
    sum2 = 0

    for i in range(len(parent1.color_ratios)):
        use_parent1 = any(start <= i < end for start, end in crossover_points)
        if use_parent1:
            ratios1.append(parent1.color_ratios[i])
            sum1 += parent1.color_ratios[i][1]
            ratios2.append(parent2.color_ratios[i])
            sum2 += parent2.color_ratios[i][1]
        else:
            ratios1.append(parent2.color_ratios[i])
            sum1 += parent2.color_ratios[i][1]
            ratios2.append(parent1.color_ratios[i])
            sum2 += parent1.color_ratios[i][1]

    if sum1 != 0:
        children.append(Individual(ratios1))
    if sum2 != 0:
        children.append(Individual(ratios2))

    return children


# One Point Crossover
# A random point is taken, and the genes of the parents are divided in two parts,
# 1) from the start of the array to and including that point,
# 2) from that excluded point to the end of the array.
# A child antisymmetric-ally receives those parts of the genes from his parents.
def one_point(parents: List[Individual]) -> List[Individual]:
    r = random.randint(0, len(parents[0].color_ratios) - 1)
    return _crossover(parents[0], parents[1], [(0, r + 1)])


def two_points(parents: List[Individual]) -> List[Individual]:
    r1 = random.randint(0, len(parents[0].color_ratios) - 1)
    r2 = random.randint(0, len(parents[0].color_ratios) - 1)
    p1 = min(r1, r2)
    p2 = max(r1, r2)
    return _crossover(parents[0], parents[1], [(0, p1 + 1), (p2 + 1, len(parents[0].color_ratios))])


def annular(parents: List[Individual]) -> List[Individual]:
    r = random.randint(0, len(parents[0].color_ratios) - 1)
    length = random.randint(0, math.ceil(len(parents[0].color_ratios) / 2))
    return _crossover(parents[0], parents[1], [(r, r + length)])


def uniform(parents: List[Individual]) -> List[Individual]:
    p = 0.5
    crossover_points = [(i, i + 1) for i in range(len(parents[0].color_ratios)) if random.random() <= p]
    return _crossover(parents[0], parents[1], crossover_points)


crossover_list = {
    "one_point": one_point,
    "two_points": two_points,
    "annular": annular,
    "uniform": uniform
}


def crossover(config_crossover: str, parents: list[Individual]) -> list[Individual]:
    return crossover_list[config_crossover](parents)
