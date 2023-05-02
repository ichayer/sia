import numpy as np

class Scaler:
    def __init__(self, range_from: (tuple[float, float] | None)=None, range_to: (tuple[float, float] | None)=None) -> None:
        if (range_from is None) != (range_to is None):
            raise Exception('Both ranges must be specified or none')
        if range_from is None:
            self.__multiplier = 1
            self.__additive = 0
        else:
            s = float(range_from[1]) - float(range_from[0])
            d = float(range_to[1]) - float(range_to[0])
            self.__multiplier = d / s
            self.__additive = -d * float(range_from[0]) / s + float(range_to[0])
    
    def scale(self, value: (float | np.ndarray[float])):
        return np.add(np.multiply(value, self.__multiplier), self.__additive)
    
    def reverse(self, value: (float | np.ndarray[float])):
        return np.divide(np.subtract(value, self.__additive), self.__multiplier)
