from src.color_tools import RGBColor, LABColor, similitude, RGBToLAB, LABToRGB, LABMix

some_green_rgb = RGBColor(55, 103, 47)
some_light_blue_rgb = RGBColor(39, 181, 187)

print("Some green ")
some_green_rgb.show()
print("Some light blue")
some_light_blue_rgb.show()

some_green_lab = RGBToLAB(some_green_rgb)
some_light_blue_lab = RGBToLAB(some_light_blue_rgb)

color_mixed_lab = LABMix(some_green_lab, some_light_blue_lab)
color_mixed_rgb = LABToRGB(color_mixed_lab)

print("Color mixed")
color_mixed_rgb.show()