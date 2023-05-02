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
    
    def from_dict(config_dict: dict):
        if config_dict is None or len(config_dict) == 0:
            return Scaler()
        if 'from' not in config_dict and 'to' not in config_dict:
            print("⚠️⚠️⚠️ WARNING: Creating empty scaler, but dictionary is not empty. Are there typos in your configuration json?")
            return Scaler()
        if ('from' in config_dict) != ('to' in config_dict):
            raise Exception('Cannot create a scaler with just one range, both from and to ranges must be specified.')
        
        range_from = (float(config_dict['from'][0]), float(config_dict['from'][1]))
        range_to = (float(config_dict['to'][0]), float(config_dict['to'][1]))
        return Scaler(range_from, range_to)
    
    def scale(self, value: (float | np.ndarray[float])):
        return np.add(np.multiply(value, self.__multiplier), self.__additive)
    
    def reverse(self, value: (float | np.ndarray[float])):
        return np.divide(np.subtract(value, self.__additive), self.__multiplier)
