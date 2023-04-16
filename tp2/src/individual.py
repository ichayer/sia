from colormath.color_conversions import convert_color
from colormath.color_objects import XYZColor, AdobeRGBColor
from colr import color
import copy
import functools


# One of the population
# Genes: proportion of each of the colors of the initial palette
# Alleles: 0-100% (float)
# Once the genes are modified, they are normalized so that together they add up to 100%.
# For example, you cannot have 60% blue and 70% red.
# Print an Individual shows the color that represents its genes, that is, what is obtained by mixing each proportion of the initial palette.
# It also shows the color that represents each gene (starter palette)
class Individual:
    xyz = XYZColor
    rgb = AdobeRGBColor

    def __init__(self, color_ratios: list[list]) -> None:
        self.color_ratios = copy.deepcopy(color_ratios)
        self.normalize()

    def __str__(self) -> str:
        r = ""
        r += f"{color('  ', fore=(0, 0, 0), back=f'{self.rgb.get_rgb_hex()}')}\t"
        for i, tupl in enumerate(self.color_ratios):
            rgb = convert_color(tupl[0], AdobeRGBColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50')
            r += f"C{i}: {color('  ', fore=(0, 0, 0), back=f'{rgb.get_rgb_hex()}')}\t {str(round(tupl[1], 2))}%\t"
        return r

    def normalize(self):
        sum_ratios = 0
        for i in range(len(self.color_ratios)):
            sum_ratios += self.color_ratios[i][1]
        for i in range(len(self.color_ratios)):
            self.color_ratios[i][1] = self.color_ratios[i][1] * 100 / sum_ratios

        sum_map = list(
            map(lambda x: (x[0].xyz_x * x[1] / 100, x[0].xyz_y * x[1] / 100, x[0].xyz_z * x[1] / 100),
                self.color_ratios))
        sum_reduce = functools.reduce(lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2]], sum_map)
        self.xyz = XYZColor(sum_reduce[0], sum_reduce[1], sum_reduce[2])
        self.rgb = convert_color(self.xyz, AdobeRGBColor, through_rgb_type=AdobeRGBColor, target_illuminant='d50')
