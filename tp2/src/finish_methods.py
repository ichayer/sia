import time


def by_time(parameters: dict, start_time: float, generation: int, distance: float) -> bool:
    if parameters["seconds"] is None:
        raise ValueError("Seconds parameter not found")

    return time.time() - start_time >= parameters["seconds"]


def by_generation(parameters: dict, start_time: float, generation: int, distance: float) -> bool:
    if parameters["generation"] is None:
        raise ValueError("Generation parameter not found")

    return generation >= parameters["generation"]


def by_distance(parameters: dict, start_time: float, generation: int, distance: float) -> bool:
    if parameters["distance"] is None:
        raise ValueError("Distance parameter not found")

    return distance <= parameters["distance"]


finish_list = {
    "by_time": by_time,
    "by_generation": by_generation,
    "by_distance": by_distance
}


def finish(config_finish: list, parameters: dict, start_time: float, generation: int, distance: float) -> bool:
    for f in config_finish:
        if finish_list[f](parameters, start_time, generation, distance):
            return True
    return False
