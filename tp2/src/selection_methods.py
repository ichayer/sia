from tp2.src.individual import Individual

def roulette():
    return 1


def elite():
    return 2


def det_tournaments():
    return 3


def prob_tournaments():
    return 4


selection_list = [roulette, elite, det_tournaments, prob_tournaments]


def selection(config_selection: int, population: list[Individual], limit: int):
    return selection_list[config_selection](population, limit)
