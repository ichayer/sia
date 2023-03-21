from .fillzone import FillzoneState

# All heuristic functions should adhere to the following function declaration:
# def heuristic_name(fz: FillzoneState) -> int

class AdmissibleHeuristics:
    def different_colors_in_game(fz: FillzoneState) -> int:
        colors = set()
        for y in range(fz.grid_size):
            for x in range(fz.grid_size):
                colors.add(fz.grid[x][y])
        return len(colors)


class NonAdmissibleHeuristics:
    def amount_of_unconquered_cells(fz: FillzoneState) -> int:
        conquered = set()
        border = [(0, 0)]
        
        while len(border) > 0:
            c = border.pop()
            if (c in conquered):
                continue
            conquered.add(c)
            for neighbor in [(c[0]-1, c[1]), (c[0]+1, c[1]), (c[0], c[1]-1), (c[0], c[1]+1)]:
                if neighbor[0] < 0 or neighbor[0] >= fz.grid_size or neighbor[1] < 0 or neighbor[1] >= fz.grid_size:
                    continue
                if fz.grid[neighbor[0]][neighbor[1]] != fz.grid[0][0]:
                    continue
                border.append(neighbor)
        return fz.grid_size * fz.grid_size - len(conquered)