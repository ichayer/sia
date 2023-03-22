import json

from src.fillzone import FillzoneState, new_fillzone_game, find_conquered_border
from src.bfs_solver import bfs_solve
from src.dfs_solver import dfs_solve
from src.greedy_solver import greedy_solve
from src.a_star_solver import a_star_solver
from src.heuristics import AdmissibleHeuristics, NonAdmissibleHeuristics

with open("config.json", "r") as f:
   config = json.load(f)

fz = new_fillzone_game(config["grid_size"], config["color_count"])
print("Initial Game:")
print(fz)

algorithm = config["algorithm"]

if algorithm == "BFS":
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
        print("Move " + str(i) + ": color " + str(move) + " (" + str(AdmissibleHeuristics.different_colors_in_game_minus_one(next)+1) + " colors left)")
        print(next)

elif algorithm == "DFS":
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
        conquered, border = find_conquered_border(next)
        print("Move " + str(i) + ": color " + str(move) + " (" + str(NonAdmissibleHeuristics.amount_of_unconquered_cells(next)) + " unconquered cells left)")
        print(str(len(conquered)) + " Conquered, " + str(len(border)) + " Bordered: " + str(list(border)))
        print(next)

elif algorithm == "GREEDY":
    print("-------------------- RUNNING GREEDY --------------------")
    result = greedy_solve(fz, AdmissibleHeuristics.different_colors_in_border)
    solution = result.solution
    print("GREEDY Solution:")
    print("Time: " + str(result.time) + "s, Nodes Expanded: " + str(result.nodes_expanded) + ", Border Nodes: " + str(result.border_nodes))

    print(str(solution) + " (" + str(len(result.solution)) + " steps)")
    print("\n")

    next = fz
    for i in range(len(solution)):
        move = solution[i]
        next = next.play_color(move)
        conquered, border = find_conquered_border(next)
        print("Move " + str(i) + ": color " + str(move) + " (" + str(NonAdmissibleHeuristics.amount_of_unconquered_cells(next)) + " unconquered cells left)")
        print(str(len(conquered)) + " Conquered, " + str(len(border)) + " Bordered: " + str(list(border)))
        print(next)

elif algorithm == "A":
    print("-------------------- RUNNING A* --------------------")
    result = a_star_solver(fz, AdmissibleHeuristics.different_colors_in_game_minus_one)
    solution = result.solution
    print("A* Solution:")
    print("Time: " + str(result.time) + "s, Nodes Expanded: " + str(result.nodes_expanded) + ", Border Nodes: " + str(result.border_nodes))

    print(str(solution) + " (" + str(len(result.solution)) + " steps)")
    print("\n")

    next = fz
    for i in range(len(solution)):
        move = solution[i]
        next = next.play_color(move)
        conquered, border = find_conquered_border(next)
        print("Move " + str(i) + ": color " + str(move) + " (" + str(NonAdmissibleHeuristics.amount_of_unconquered_cells(next)) + " unconquered cells left)")
        print(str(len(conquered)) + " Conquered, " + str(len(border)) + " Bordered: " + str(list(border)))
        print(next)

else:
    print("Invalid algorithm: " + algorithm + ". Valid algorithms are: BFS, DFS, GREEDY, A")