# TP1 

### Requisitos:

* Python 3.10.X
* Los siguientes paquetes instalables mediante `pip`:

```py
pip install matplotlib
pip install numpy
```

### Creación del Estado Inicial:

Para poder ejecutar los distintos algoritmos primero se debe crear un nuevo juego de Fillzone.
Para ello se utiliza el método `new_fillzone_game(grid_size, color_count)` que retorna un `FillzoneState`.

```py
from src.fillzone import new_fillzone_game
fz = new_fillzone_game(7, 5)
```

En este sentido `fz` será el estado inicial del juego.

### Métodos de Búsqueda:

Para los diferentes algoritmos tenemos métodos _algorithm_solve()_, que reciben un `FillzoneState` y devuelven un `FillzoneSolveResult`. Este objeto contiene la solución (`.solution`), cantidad de nodos expandidos (`.nodes_expanded`), cantidad de nodos frontera al finalizar (`.border_nodes`), y tiempo de ejecución en segundos (`.time`).

```py
from src.bfs_solver import bfs_solve
from src.dfs_solver import dfs_solve

result_bfs = bfs_solve(fz)
result_dfs = dfs_solve(fz)
```

En cuanto a las heurísticas, están implementadas como funciones estáticas en las classes AdmissibleHeuristics y NonAdmissibleHeuristics. Los algoritmos de búsqueda informados reciben la heurística a utilizar como segundo parámetro:

```py
from src.greedy_solver import greedy_solve
from src.a_star_solver import a_star_solve
from src.heuristics import AdmissibleHeuristics, NonAdmissibleHeuristics

result_greedy = greedy_solve(fz, AdmissibleHeuristic.different_colors_in_game_minus_one)
result_astar = a_star_solve(fz, NonAdmissibleHeuristic.amount_of_unconquered_cells)
```
