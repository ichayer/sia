import functools
import copy
from colr import color
from colormath.color_objects import XYZColor, AdobeRGBColor
from colormath.color_conversions import convert_color
from .color_tools import MyXYZColor

#one of the population
class Individual:
    def __init__(self, color_ratios: list[list[MyXYZColor, float]]) -> None:
        self.color_ratios = copy.deepcopy(color_ratios)
    
    def __str__(self) -> str:
        r = ""
        for i, tuple in enumerate(self.color_ratios):
            xyz = XYZColor(tuple[0].color.xyz_x, tuple[0].color.xyz_y, tuple[0].color.xyz_z)
            rgb = convert_color(xyz, AdobeRGBColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50')
            r += f"C{i}: {color('  ', fore=(0, 0, 0), back=f'{rgb.get_rgb_hex()}')}\t {tuple[1]}%\t"
            #r += f"{tuple[0]}({tuple[1]}) \t{color('  ', fore=(0, 0, 0), back=f'{rgb.get_rgb_hex()}')}\n"
        return r
    
    def show(self):
        sum = functools.reduce(lambda x,y: (x[0].xyz_x + x[1].xyz_x, x[0].xyz_y + x[1].xyz_y, x[0].xyz_y + x[1].xyz_y), self.color_ratios)
        xyz = XYZColor(sum[0], sum[1], sum[2])
        rgb = convert_color(xyz, AdobeRGBColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50')
        print(color('  ', fore=(0, 0, 0), back=f'{rgb.get_rgb_hex()}'))

from .crossover_methods import crossover
from .selection_methods import selection

#parameters are received as RGB for simplicity but we saved as XYZ
class ColorGeneticAlgorithm:
    actual_gen = []
    
    def __init__(self, color_set: list[AdobeRGBColor], color_target: AdobeRGBColor, selection: int, crossover: int, mutation: int, finish: int, poblation: int) -> None:
        self.color_set = list(map(lambda x: convert_color(x, XYZColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50'), color_set))
        self.color_target = Individual({(convert_color(color_target, XYZColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50'),100)})
        self.selection_method = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.finish_method = finish
        self.poblation = poblation
        self.__gen0()
        
    def __gen0(self) -> None:
        ratios = []
        
        for color in self.color_set:
            ratios.append([MyXYZColor(color), 0])
            
        ratios[0][1] = 100
        self.actual_gen.append(Individual(ratios))
        
        for i in range(1, len(self.color_set)):
            ratios[i-1][1] = 0
            ratios[i][1] = 100
            self.actual_gen.append(Individual(ratios))
        
        for i, ind in enumerate(self.actual_gen):
            print(f"Individuo {i}: {ind}")
    
    def start(self) -> None:
        print("Starting the algorithm...")
        new_gen = []
        
        for i, ind in enumerate(self.actual_gen):
            l = crossover(self.selection_method, [self.actual_gen[i], self.actual_gen[(i+1)%len(self.actual_gen)]])
            new_gen.append(l[0])
            new_gen.append(l[1])
        
        for i, ind in enumerate(new_gen):
            print(f"Individuo {i}: {ind}")