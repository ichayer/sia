import json
from colormath.color_objects import AdobeRGBColor
from src.color_ga import ColorGeneticAlgorithm

with open("config.json", "r") as f:
    config = json.load(f)

red = AdobeRGBColor(255,0,0,True)
green = AdobeRGBColor(0,255,0,True)
blue = AdobeRGBColor(0,0,255,True)
target = AdobeRGBColor(12,232,156,True)

ga = ColorGeneticAlgorithm([red, green, blue], target, config["selection"], config["crossover"], config["mutation"], config["finish"], config["poblation"])

ga.start()