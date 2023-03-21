from .fillzone import FillzoneState, FillzoneSolveResult
from time import perf_counter

def greedy_solve(fz: FillzoneState) -> FillzoneSolveResult:
    return FillzoneSolveResult(None, 0, 0, 0)