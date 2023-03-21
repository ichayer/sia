import heapq

from .fillzone import FillzoneState, FillzoneSolveResult
from time import perf_counter


class FillzoneAStarNode:
    def __init__(self, state: FillzoneState, g_value: int, h_value: int, parent: 'FillzoneAStarNode' = None):
        self.state = state
        self.g_value = g_value
        self.h_value = h_value
        self.parent = parent

    def f_value(self):
        return self.g_value + self.h_value

    def __lt__(self, other):
        if self.f_value() == other.f_value:
            return self.h_value < other.h_value

        return self.f_value() < other.f_value()

def a_star_solver(state: FillzoneState, heuristic: callable(FillzoneState)) -> FillzoneSolveResult:
    states_explored = set()
    border = [(heuristic(state), FillzoneAStarNode(state, 0, 0, None))]
    time_start = perf_counter()

    while len(border) > 0:
        # node_current is always the node with less f(n)
        _, node_current = heapq.heappop(border)
        state_current = node_current.state

        # Fill-zone solved :)
        if state_current.is_solved():
            return get_fill_zone_zone_result(border, node_current, states_explored, time_start)

        # State already explored
        if state_current in states_explored:
            continue

        states_explored.add(state_current)

        for color in range(state.color_count):
            if color == state_current.get_target_color():
                continue

            state_next = state_current.play_color(color)
            h_value = heuristic(state_next)

            # since fill-zone does not have a transition state cost, we define it as the number of steps
            # taken to reach the current state from the initial state.
            g_value = node_current.g_value + 1
            node_next = FillzoneAStarNode(state_next, g_value, h_value, node_current)
            f_value = node_next.f_value()

            heapq.heappush(border, (f_value, node_next))

    # All fill-zone games are solvable. This should never happen.
    return FillzoneSolveResult([], len(states_explored), len(border), perf_counter() - time_start)


def get_fill_zone_zone_result(border, node_current, states_explored, time_start) -> FillzoneSolveResult:

    result = []
    while node_current is not None:
        result.append(node_current.state.grid[0][0])
        node_current = node_current.parent
    result.reverse()
    return FillzoneSolveResult(result, len(states_explored), len(border), perf_counter() - time_start)
