import math
from colormath.color_objects import XYZColor


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


def similitude(c1: MyXYZColor, c2: MyXYZColor) -> float:
    return math.sqrt(
        c1.color.xyz_x * c2.color.xyz_x + c1.color.xyz_y * c2.color.xyz_y + c1.color.xyz_z * c2.color.xyz_z)
