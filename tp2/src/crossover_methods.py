from enum import Enum
import random
from .color_ga import Individual

class Crossover(Enum):
	ONE_POINT = 1
	TWO_POINTS = 2
	ANNULAR = 3
	UNIFORM = 4

def one_point(parents: list[Individual]) -> list[Individual]:
    r = random.randint(0, len(parents[0].color_ratios)-1)
    print(r)
    l = []
    ratios1 = []
    ratios2 = []
    for i in range(len(parents[0].color_ratios)):
        if(i <= r):
            ratios1.append(parents[0].color_ratios[i])
            ratios2.append(parents[1].color_ratios[i])
        else:
            ratios1.append(parents[1].color_ratios[i])
            ratios2.append(parents[0].color_ratios[i])
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