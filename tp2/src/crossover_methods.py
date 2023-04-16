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
    l = []
    r = random.randint(0, len(parents[0].color_ratios) - 1)
    ratios1 = []
    ratios2 = []
    sum1 = 0
    sum2 = 0
    for i in range(len(parents[0].color_ratios)):
        if i <= r:
            ratios1.append(parents[0].color_ratios[i])
            sum1 += parents[0].color_ratios[i][1]
            ratios2.append(parents[1].color_ratios[i])
            sum2 += parents[1].color_ratios[i][1]
        else:
            ratios1.append(parents[1].color_ratios[i])
            sum1 += parents[1].color_ratios[i][1]
            ratios2.append(parents[0].color_ratios[i])
            sum2 += parents[0].color_ratios[i][1]

    if sum1 != 0:
        l.append(Individual(ratios1))
    if sum2 != 0:
        l.append(Individual(ratios2))
    return l


def two_points(parents: list[Individual]) -> list[Individual]:
    return 2


def annular(parents: list[Individual]) -> list[Individual]:
    return 3


def uniform(parents: list[Individual]) -> list[Individual]:
    return 4


crossover_list = [one_point, two_points, annular, uniform]


def crossover(config_crossover: int, parents: list[Individual]) -> list[Individual]:
    return crossover_list[config_crossover](parents)
