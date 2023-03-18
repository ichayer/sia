from src.fillzone import FillzoneState, new_fillzone_game
from src.bfs_solver import bfs_solve

fz = new_fillzone_game(5, 4)
print("Initial Game:")
print(fz)

solution = bfs_solve(fz)
print("Solution:")
print(solution)
print("\n")

for i in range(len(solution)):
    move = solution[i]
    fz = fz.play_color(move)
    print("Move " + i.__str__() + ": color " + move.__str__())
    print(fz)

#for i in range(fz.color_count):
#    print("\nPlaying color " + i.__str__())
#    fz = fz.play_color(i)
#    print(fz)