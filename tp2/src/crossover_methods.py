import math
import random
from .individual import Individual

# Crossover functions return one or two children. Only one child if one of them has all genes at 0% ("Dead individual")
# To do this, the ratios of the colors are added in a variable as they are iterated.

# One Point Crossover
# A random point is taken, and the genes of the parents are divided in two parts,
# 1) from the start of the array to and including that point,
# 2) from that excluded point to the end of the array.
# A child antisymmetric-ally receives those parts of the genes from his parents.
def one_point(parents: list[Individual]) -> list[Individual]:
    children = []
    r = random.randint(0, len(parents[0].color_ratios) - 1)
    ratios1 = []
    ratios2 = []
    sum1 = 0
    sum2 = 0
    parent1 = parents[0]
    parent2 = parents[1]
    for i in range(len(parent1.color_ratios)):
        if i <= r:
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


def two_points(parents: list[Individual]) -> list[Individual]:
    children = []
    r1 = random.randint(0, len(parents[0].color_ratios) - 1)
    r2 = random.randint(0, len(parents[0].color_ratios) - 1)
    p1 = r1 if r1 >= r2 else r2
    p2 = r1 if r1 < r2 else r2
    ratios1 = []
    ratios2 = []
    sum1 = 0
    sum2 = 0
    parent1 = parents[0]
    parent2 = parents[1]
    for i in range(len(parent1.color_ratios)):
        if i <= p1 or i > p2:
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


def annular(parents: list[Individual]) -> list[Individual]:
    children = []
    r = random.randint(0, len(parents[0].color_ratios) - 1)
    length = random.randint(0, math.ceil(len(parents[0].color_ratios) / 2))
    ratios1 = []
    ratios2 = []
    sum1 = 0
    sum2 = 0
    parent1 = parents[0]
    parent2 = parents[1]
    for i in range(len(parent1.color_ratios)):
        circular_position = (i - r + len(parent1.color_ratios)) % len(parent1.color_ratios)
        if circular_position < length:
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


def uniform(parents: list[Individual]) -> list[Individual]:
    return 4


crossover_list = {
    "one_point": one_point,
    "two_points": two_points,
    "annular": annular,
    "uniform": uniform
}


def crossover(config_crossover: str, parents: list[Individual]) -> list[Individual]:
    return crossover_list[config_crossover](parents)
