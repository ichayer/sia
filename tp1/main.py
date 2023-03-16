from src.fillzone import FillzoneState, new_fillzone_game

fz = new_fillzone_game(5, 4)
print("Initial Game:")
print(fz)

for i in range(5):
    print("\nPlaying color " + i.__str__())
    fz = fz.play_color(i)
    print(fz)