import json
from colormath.color_objects import AdobeRGBColor
from src.color_ga import ColorGeneticAlgorithm

with open("config.json", "r") as f:
    config = json.load(f)

red = AdobeRGBColor(255, 0, 0, True)
green = AdobeRGBColor(0, 255, 0, True)
blue = AdobeRGBColor(0, 0, 255, True)
yellow = AdobeRGBColor(255, 255, 0, True)
pink = AdobeRGBColor(255, 0, 255, True)
cian = AdobeRGBColor(0, 255, 255, True)
target = AdobeRGBColor(135, 255, 12, True)

ga = ColorGeneticAlgorithm([red, green, blue, yellow, pink, cian], target, config["selection"], config["crossover"], config["mutation"],
                           config["finish"], config["population"])


ga.start()