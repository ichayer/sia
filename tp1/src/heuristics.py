from .fillzone import FillzoneState, find_conquered_border

# All heuristic functions should adhere to the following function declaration:
# def heuristic_name(fz: FillzoneState) -> int

class AdmissibleHeuristics:
    def different_colors_in_game(fz: FillzoneState) -> int:
        colors = set()
        for y in range(fz.grid_size):
            for x in range(fz.grid_size):
                colors.add(fz.grid[x][y])
        return len(colors)
    
    
    def different_colors_in_border(fz: FillzoneState) -> int:
        conquered, border = find_conquered_border(fz)
        colors = set()
        for cell in border:
            colors.add(fz.grid[cell[0]][cell[1]])
        return len(colors)


class NonAdmissibleHeuristics:
    def amount_of_unconquered_cells(fz: FillzoneState) -> int:
        conquered, border = find_conquered_border(fz)
        return fz.grid_size * fz.grid_size - len(conquered)