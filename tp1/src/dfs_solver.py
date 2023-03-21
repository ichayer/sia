from .fillzone import FillzoneState, FillzoneSolveResult
from time import perf_counter

def dfs_solve(fz: FillzoneState) -> FillzoneSolveResult:
    explored = set()
    border = [(fz, None)]
    time_start = perf_counter()


    while len(border) > 0:
        current = border.pop()
        
        if current[0].is_solved():
            result = []
            while current[1] != None:
                result.append(current[0].grid[0][0])
                current = current[1]
            result.reverse()
            return FillzoneSolveResult(result, len(explored), len(border), perf_counter() - time_start)
        
        if current[0] in explored:
            continue
        
        explored.add(current[0])
        for color in range(fz.color_count):
            if color == current[0].grid[0][0]:
                continue
            next = current[0].play_color(color)
            border.append((next, current))
        
    
    # All fillzone games are solvable. This should never happen.
    return FillzoneSolveResult(None, len(explored), len(border), perf_counter() - time_start)