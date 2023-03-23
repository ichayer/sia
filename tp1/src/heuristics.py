from .fillzone import FillzoneState, find_conquered_border
from collections import deque

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
        
    def buscaminas_distance(fz: FillzoneState) -> int:
        explored = set()
        border = deque([(0,0,0)])
        max = 0
        
        while(len(border) > 0):
            c = border.pop()
            celd = (c[0], c[1])
            from_color = fz.grid[c[0]][c[1]]
            if celd in explored:
                continue
            explored.add(celd)
            for neighbor in [(c[0]-1, c[1]), (c[0]+1, c[1]), (c[0], c[1]-1), (c[0], c[1]+1)]:
                if neighbor[0] < 0 or neighbor[0] >= fz.grid_size or neighbor[1] < 0 or neighbor[1] >= fz.grid_size:
                    continue
                if fz.grid[neighbor[0]][neighbor[1]] in explored:
                    continue
                if fz.grid[neighbor[0]][neighbor[1]] == from_color:
                    border.append((neighbor[0],neighbor[1], c[2]))
                    continue
                if fz.grid[neighbor[0]][neighbor[1]] != from_color:
                    border.appendleft((neighbor[0],neighbor[1], c[2]+1))
            max = c[2]
        return max

class NonAdmissibleHeuristics:
    def amount_of_unconquered_cells(fz: FillzoneState) -> int:
        conquered, border = find_conquered_border(fz)
        return fz.grid_size * fz.grid_size - len(conquered)
