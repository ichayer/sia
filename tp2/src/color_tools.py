import math
from colormath.color_objects import XYZColor, LabColor
from colormath.color_conversions import convert_color


# The similarity between two colors is calculated in the LAB space because it is the most accurate to the human eye.
def similitude(c1: XYZColor, c2: XYZColor) -> float:
    lab1 = convert_color(c1, LabColor, through_rgb_type=LabColor, target_illuminant='d50')
    lab2 = convert_color(c2, LabColor, through_rgb_type=LabColor, target_illuminant='d50')
    return math.sqrt(
        (lab1.lab_l - lab2.lab_l) * (lab1.lab_l - lab2.lab_l) + (lab1.lab_a - lab2.lab_a) * (
                lab1.lab_a - lab2.lab_a) + (lab1.lab_b - lab2.lab_b) * (lab1.lab_b - lab2.lab_b))
