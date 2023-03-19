from src.fillzone import FillzoneState, new_fillzone_game
from src.bfs_solver import bfs_solve
from src.dfs_solver import dfs_solve

fz = new_fillzone_game(8, 5)
print("Initial Game:")
print(fz)

print("-------------------- RUNNING BFS --------------------")
result = bfs_solve(fz)
solution = result.solution
print("BFS Solution:")
print("Time: " + result.time.__str__() + "s, Nodes Expanded: " + result.nodes_expanded.__str__() + ", Border Nodes: " + result.border_nodes.__str__())

print(solution.__str__() + " (" + len(result.solution).__str__() + " steps)")
print("\n")

next = fz
for i in range(len(solution)):
    move = solution[i]
    next = next.play_color(move)
    print("Move " + i.__str__() + ": color " + move.__str__())
    print(next)

print("-------------------- RUNNING DFS --------------------")
result = dfs_solve(fz)
solution = result.solution
print("DFS Solution:")
print("Time: " + result.time.__str__() + "s, Nodes Expanded: " + result.nodes_expanded.__str__() + ", Border Nodes: " + result.border_nodes.__str__())

print(solution.__str__() + " (" + len(result.solution).__str__() + " steps)")
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