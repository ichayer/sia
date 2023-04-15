from enum import Enum
from .color_ga import Individual

class Selection(Enum):
	ROULETE = 1
	ELITE = 2
	DET_TOURNAMENTS = 3
	PROB_TOURNAMENTS = 4

def roulete():
    return 1

def elite():
    return 2

def det_tournaments():
    return 3

def prob_tournaments():
    return 4

selection_list = [roulete, elite, det_tournaments, prob_tournaments]

def selection(config_selection: int, poblation: list[Individual]):
    return selection_list[config_selection](poblation)