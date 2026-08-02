import heapq
from typing import List, Tuple, Dict, Any
from src.client.models import HexType

HEX_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

def astar_hex_path(
        start: Tuple[int, int],
        target: Tuple[int, int],
        max_op: int,
        game_state
) -> List[Dict[str, int]]:
    if start == target:
        return []

    counter = 0
    open_set = []
    heapq.heappush(open_set, (0, counter, 0, start, []))
    visited = {}

    best_path = []

    while open_set:
        _, _, current_cost, current, path = heapq.heappop(open_set)

        if current in visited and visited[current] <= current_cost:
            continue
        visited[current] = current_cost

        if current == target:
            best_path = path
            break

        cq, cr = current
        for dq, dr in HEX_NEIGHBORS:
            nq, nr = cq + dq, cr + dr

            if game_state.is_border(nq, nr):
                continue

            th = game_state.get_hex(nq, nr)

            # Проверка проходимости гекса
            if th is not None:
                hex_type = getattr(th, "type", None)
                if hex_type == 5 or hex_type == HexType.STONE:
                    continue
                step_cost = getattr(th, "cost", 1)
            else:
                # Если клетка скрыта туманом войны, считаем её обычной сушей
                step_cost = 1

            if step_cost >= 999:
                continue

            new_cost = current_cost + step_cost

            if new_cost <= max_op:
                new_path = path + [{"q": nq, "r": nr}]
                h_score = (abs(nq - target[0]) + abs(nr - target[1]) + abs((nq + nr) - (target[0] + target[1]))) // 2
                f_score = new_cost + h_score

                counter += 1
                heapq.heappush(open_set, (f_score, counter, new_cost, (nq, nr), new_path))

    return best_path
