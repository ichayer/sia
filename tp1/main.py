from src.fillzone import FillzoneState, new_fillzone_game
from src.bfs_solver import bfs_solve
from src.dfs_solver import dfs_solve
from src.heuristics import AdmissibleHeuristics, NonAdmissibleHeuristics

fz = new_fillzone_game(8, 5)
print("Initial Game:")
print(fz)

print("-------------------- RUNNING BFS --------------------")
result = bfs_solve(fz)
solution = result.solution
print("BFS Solution:")
print("Time: " + str(result.time) + "s, Nodes Expanded: " + str(result.nodes_expanded) + ", Border Nodes: " + str(result.border_nodes))

print(str(solution) + " (" + str(len(result.solution)) + " steps)")
print("\n")

next = fz
for i in range(len(solution)):
    move = solution[i]
    next = next.play_color(move)
    print("Move " + str(i) + ": color " + str(move))
    print(next)

print("-------------------- RUNNING DFS --------------------")
result = dfs_solve(fz)
solution = result.solution
print("DFS Solution:")
print("Time: " + str(result.time) + "s, Nodes Expanded: " + str(result.nodes_expanded) + ", Border Nodes: " + str(result.border_nodes))

print(str(solution) + " (" + str(len(result.solution)) + " steps)")
print("\n")

next = fz
for i in range(len(solution)):
    move = solution[i]
    next = next.play_color(move)
    print("Move " + str(i) + ": color " + str(move))
    print(next)


#for i in range(fz.color_count):
#    print("\nPlaying color " + i.__str__())
#    fz = fz.play_color(i)
#    print(fz)