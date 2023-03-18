from .fillzone import FillzoneState

def dfs_solve(fz: FillzoneState) -> list[int]:
    explored = set()
    border = [(fz, None)]
    
    while len(border) > 0:
        current = border.pop()
        
        if current[0].is_solved():
            result = []
            while current[1] != None:
                result.append(current[0].grid[0][0])
                current = current[1]
            result.reverse()
            return result
        
        if current[0] in explored:
            continue
        
        explored.add(current[0])
        for nextmove in range(fz.color_count):
            if nextmove == current[0].grid[0][0]:
                continue
            next = current[0].play_color(nextmove)
            border.append((next, current))
        
    
    # All fillzone games are solvable. This should never happen.
    return None