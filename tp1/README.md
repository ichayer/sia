# TP1 

### Requisitos:

* Python 3.10.X

### Creación del Estado Inicial:

Para poder ejecutar los distintos algoritmos primero se debe crear un nuevo juego de Fillzone.
Para ello se utiliza el método _new_fillzone_game_ que retorna un _FillzoneState_.

**fz = new_fillzone_game(grid_size, color_count)**

En este sentido fz será el estado inicial del juego.

### Métodos de Búsqueda:

Para los diferentes algoritmos tenemos métodos _algorithm_solve_, que reciben un estado fillzone y devuelven un resultado. Cabe aclarar que los algoritmos de búsqueda informados reciben la heurística pertinente como segundo parámetro.

**result = bfs_solve(fz), dfs_solve(fz), greedy_solve(fz, heuristic),  a_star_solve(fz, heuristic)**

Luego, el resultado contiene el número de nodos explorados (_result.nodes_expanded_), de nodos borde (_result.border_nodes_), el tiempo de ejecución (_result.time_), y el paso por paso para encontrar la solución (_result.solution[]_).