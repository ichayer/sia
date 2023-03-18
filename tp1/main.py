from src.fillzone import FillzoneState, new_fillzone_game
from src.bfs_solver import bfs_solve
from src.dfs_solver import dfs_solve

fz = new_fillzone_game(5, 4)
print("Initial Game:")
print(fz)

print("-------------------- RUNNING BFS --------------------")
solution = bfs_solve(fz)
print("BFS Solution:")
print(solution)
print("\n")

next = fz
for i in range(len(solution)):
    move = solution[i]
    next = next.play_color(move)
    print("Move " + i.__str__() + ": color " + move.__str__())
    print(next)

print("-------------------- RUNNING DFS --------------------")
solution = dfs_solve(fz)
print("DFS Solution:")
print(solution)
print("\n")

next = fz
for i in range(len(solution)):
    move = solution[i]
    next = next.play_color(move)
    print("Move " + i.__str__() + ": color " + move.__str__())
    print(next)

#for i in range(fz.color_count):
#    print("\nPlaying color " + i.__str__())
#    fz = fz.play_color(i)
#    print(fz)