import math
from colormath.color_objects import XYZColor, LabColor
from colormath.color_conversions import convert_color


class MyXYZColor:

    def __init__(self, color: XYZColor) -> None:
        self.color = color

    def __hash__(self) -> int:
        return hash(self.color.xyz_x, self.color.xyz_y, self.color.xyz_z)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, type(self)): return NotImplemented
        return math.isclose(self.color.xyz_x, __value.color.xyz_x) \
            and math.isclose(self.color.xyz_y, __value.color.xyz_y) \
            and math.isclose(self.color.xyz_z, __value.color.xyz_z)

    def __str__(self):
        return f"x:{self.color.xyz_x} \t y:{self.color.xyz_y} \t z:{self.color.xyz_z}"

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)


def similitude(c1: XYZColor, c2: XYZColor) -> float:
    lab1 = convert_color(c1, LabColor, through_rgb_type=LabColor, target_illuminant='d50')
    lab2 = convert_color(c2, LabColor, through_rgb_type=LabColor, target_illuminant='d50')
    return math.sqrt(
        (lab1.lab_l - lab2.lab_l)*(lab1.lab_l - lab2.lab_l) + (lab1.lab_a - lab2.lab_a)*(lab1.lab_a - lab2.lab_a) + (lab1.lab_b - lab2.lab_b)*(lab1.lab_b - lab2.lab_b))


    # return 1 - math.sqrt((c1.xyz_x - c2.xyz_x)*(c1.xyz_x - c2.xyz_x) + (c1.xyz_y - c2.xyz_y)*(c1.xyz_y - c2.xyz_y) + (c1.xyz_z - c2.xyz_z)*(c1.xyz_z - c2.xyz_z))
