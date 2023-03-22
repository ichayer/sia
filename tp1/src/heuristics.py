from .fillzone import FillzoneState, find_conquered_border

# All heuristic functions should adhere to the following function declaration:
# def heuristic_name(fz: FillzoneState) -> int

class AdmissibleHeuristics:
    def different_colors_in_game_minus_one(fz: FillzoneState) -> int:
        colors = set()
        for y in range(fz.grid_size):
            for x in range(fz.grid_size):
                colors.add(fz.grid[x][y])
        return len(colors) - 1
    
    
    def different_colors_in_border(fz: FillzoneState) -> int:
        conquered, border = find_conquered_border(fz)
        colors = set()
        for cell in border:
            colors.add(fz.grid[cell[0]][cell[1]])
        return len(colors)
      
    def combination_of_heuristics(fz: FillzoneState, *heuristics) -> int:
      ret = []
      for heuristic in heuristics:
        ret.append(heuristic(fz))
        
      return max(ret)


class NonAdmissibleHeuristics:
    def amount_of_unconquered_cells(fz: FillzoneState) -> int:
        conquered, border = find_conquered_border(fz)
        return fz.grid_size * fz.grid_size - len(conquered)
