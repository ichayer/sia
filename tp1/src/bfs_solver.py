from collections import deque
from .fillzone import FillzoneState, FillzoneSolveResult
from time import perf_counter

def bfs_solve(fz: FillzoneState) -> FillzoneSolveResult:
    explored = set()
    border = deque([(fz, None)])
    time_start = perf_counter()
    
    while len(border) > 0:
        current = border.popleft()
        
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
        for nextmove in range(fz.color_count):
            if nextmove == current[0].grid[0][0]:
                continue
            next = current[0].play_color(nextmove)
            border.append((next, current))
        
    
    # All fillzone games are solvable. This should never happen.
    return FillzoneSolveResult(None, len(explored), len(border), perf_counter() - time_start)