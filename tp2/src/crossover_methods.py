import random
from tp2.src.individual import Individual


# Bc performance, individuals are not returned, but their genes

def one_point(parents: list[Individual]) -> list[Individual]:
    l = []
    sum2 = 0
    sum1 = 0
    ratios1 = []
    ratios2 = []
    while sum1 == 0 or sum2 == 0:
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

    l.append(Individual(ratios1))
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
