from random import choice
from typing import *

import random

from client import *
from gamestate import *


class PathFinder:
    @staticmethod
    def hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return (abs(a[0] - b[0]) + abs(a[0] + a[1] - b[0] - b[1]) + abs(a[1] - b[1])) // 2

    @staticmethod
    def get_neighbors(q: int, r: int) -> List[Tuple[int, int]]:
        return PathFinder.hex_in_area((q, r), is_small_strict=True)

    @staticmethod
    def hex_in_area(
            center: Tuple[int, int],
            small_radius: int = 0,
            big_radius: int = 1,
            is_small_strict: bool = False,
            is_big_strict: bool = False) -> List[Tuple[int, int]]:
        """
        Возвращает гексы в заданной области вокруг центрального гекса.

        Параметры:
            center: (q, r) координаты центрального гекса
            small_radius: внутренний радиус области (по умолчанию 0)
            big_radius: внешний радиус области (по умолчанию 1)
            is_small_strict: если True, исключает гексы на внутреннем радиусе
            is_big_strict: если True, исключает гексы на внешнем радиусе

        Возвращает:
            Список кортежей (q, r) гексов в заданной области

        Примеры:
            hex_in_area((3, 2)) -> возвращает центральный гекс и всех соседей (радиус 0-1)
            hex_in_area((3, 2), big_radius=2) -> радиус 0-2
            hex_in_area((3, 2), small_radius=1, big_radius=2, is_small_strict=True) -> только гексы на расстоянии 2
        """

        if big_radius < small_radius:
            raise ValueError("Больший радиус должен быть >= малый радиус")

        center_q, center_r = center
        hexes = list()

        for q in range(-big_radius, big_radius + 1):
            for r in range(max(-big_radius, - big_radius), min(big_radius, + big_radius) + 1):
                distance = PathFinder.hex_distance((0, 0), (q, r))

                if (distance > small_radius or (not is_small_strict and distance == small_radius)) and \
                        (distance < big_radius or (not is_big_strict and distance == big_radius)):
                    hex_q = center_q + q
                    hex_r = center_r + r
                    hexes.append((hex_q, hex_r))

        return hexes

    @staticmethod
    def heuristic(current: Hex, target: Tuple[int, int]) -> float:
        return PathFinder.hex_distance(current.hex, target) + (3.2 if current.type == HexType.ACID else 0)


class Strategy:
    def __init__(self, client: DatsPulseClient):
        self.client = client
        self.memory = None

    def make_turn(self, state: GameState):
        moves = list()
        if state.memory is not None:
            self.memory = state.memory

        for ant in state.ants:
            if ant.type == AntType.SCOUT:
                path = self.scout_strategy(ant, state)
            elif ant.type == AntType.SOLDIER:
                path = self.soldier_strategy(ant, state)
            else:  # AntType.WORKER
                path = self.worker_strategy(ant, state)
            print(self.memory.roles.get(ant.id, AntRole.SIMPLE), path[-1] if path else ant.hex)

            if path:
                path.pop(0)
                moves.append({
                    'ant': ant.id,
                    'path': [{'q': q, 'r': r} for q, r in path]
                })

        if moves:
            self.client.send_moves(moves)

    def recall_all_soldiers_to_base(self, state: GameState):
        base_q, base_r = state.spot['q'], state.spot['r']
        moves = list()

        for ant in state.ants:
            if ant.type == AntType.SOLDIER:
                path = self.find_path(ant.q, ant.r, base_q, base_r, state, subject=ant)
                if path:
                    moves.append({
                        'ant': ant.id,
                        'path': [{'q': q, 'r': r} for q, r in path]
                    })

        if moves:
            self.client.send_moves(moves)

    def recall_all_workers_to_base(self, state: GameState):
        base_q, base_r = state.spot['q'], state.spot['r']
        moves = list()

        for ant in state.ants:
            if ant.type == AntType.WORKER:
                path = self.find_path(ant.q, ant.r, ANTS_PROPERTY[ant.type]['speed'], base_q, base_r, state)
                if path:
                    moves.append({
                        'ant': ant.id,
                        'path': [{'q': q, 'r': r} for q, r in path]
                    })

        if moves:
            self.client.send_moves(moves)

    def stay_in_range(self, ant, r_small: int, r_big: int, spacing: int, center, state):
        radius = r_small - 1
        while radius < r_big + 1:
            radius += 1

            # Кольцевые позиции
            ring_positions = PathFinder.hex_in_area(center, small_radius=radius, big_radius=radius)
            ring_positions = ring_positions[::spacing]
            ring_positions = list(filter(lambda h: not state.get_hex(*h) or state.get_hex(*h).type not in (
                HexType.ANTHILL, HexType.STONE, HexType.ACID), ring_positions))

            if ant.hex in ring_positions:
                return list()

            # Позиции, которые уже заняты
            reserved = set()
            for a in state.ants:
                if a.type == AntType.SOLDIER:
                    reserved.add((a.q, a.r))

            # Свободные позиции
            free_positions = {pos for pos in ring_positions if pos not in reserved}

            if free_positions:
                nearest_free = min(free_positions, key=lambda pos: PathFinder.hex_distance(ant.hex, pos))
                return self.find_path(ant.q, ant.r, nearest_free[0], nearest_free[1], state, subject=ant)
        return list()

    def scout_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        role = self.memory.roles.get(ant.id, AntRole.SIMPLE)
        if ant.type != AntType.SCOUT:
            return [ant.hex, random.choice(PathFinder.get_neighbors(*ant.hex))]

        if role == AntRole.SIMPLE:
            role = choice((AntRole.SCOUT, AntRole.DEFENDER))
            self.memory.roles[ant.id] = role

        if role == AntRole.SCOUT:
            hexes = set(self.memory.map.keys()) - set(map(lambda h: h.hex, state.map))
            mx = max(set(self.memory.map.keys()))
            mn = min(set(self.memory.map.keys()))
            hexes |= {(mx[0] + 1, mx[1] + 1), (mn[0] - 1, mn[0] - 1)}
            hex = min(hexes, key=lambda h: PathFinder.hex_distance(h, ant.hex))
            return self.find_path(*ant.hex, *hex, state, subject=ant)

        if role == AntRole.DEFENDER:
            home_q, home_r = state.spot['q'], state.spot['r']
            home_hex = (home_q, home_r)
            return self.stay_in_range(ant, 9, 25, 6, center=home_hex, state=state)

        return list() if state.get_hex(*ant.hex).type not in (HexType.ANTHILL, HexType.ACID) \
            else [ant.hex, random.choice(PathFinder.get_neighbors(*ant.hex))]

    def soldier_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        role = self.memory.roles.get(ant.id, AntRole.SIMPLE)
        home_q, home_r = state.spot['q'], state.spot['r']
        home_hex = (home_q, home_r)
        ant_hex = (ant.q, ant.r)

        if role == AntRole.SIMPLE:
            role = choice((AntRole.DEFENDER, AntRole.ARMY, AntRole.ESCORT))
            self.memory.roles[ant.id] = role

        if role == AntRole.ARMY and self.memory.have_attack:
            return self.stay_in_range(ant, 1, 5, 1, center=self.memory.enemy_hill.hex, state=state)

        for enemy in state.enemies:
            if PathFinder.hex_distance(enemy.hex, ant.hex) < 5:
                return self.find_path(ant.q, ant.r, *enemy.hex, state, subject=ant)

        if role == AntRole.DEFENDER:
            return self.stay_in_range(ant, 7, 12, 3, center=home_hex, state=state)

        if role == AntRole.ESCORT or role == AntRole.ARMY and not self.memory.have_attack:
            # Сопровождаем рабочих, идущих от базы
            workers_on_mission = {
                a for a in state.ants
                if a.type == AntType.WORKER and PathFinder.hex_distance((a.q, a.r), home_hex) > 3
            }
            if workers_on_mission:
                nearest_worker = min(workers_on_mission, key=lambda w: PathFinder.hex_distance(ant_hex, (w.q, w.r)))
                return self.find_path(ant.q, ant.r, nearest_worker.q, nearest_worker.r, state, subject=ant)

        return list() if state.get_hex(*ant_hex).type not in (HexType.ANTHILL, HexType.ACID)\
            else [ant.hex, random.choice(PathFinder.get_neighbors(*ant_hex))]

    def worker_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        home_q, home_r = state.spot['q'], state.spot['r']
        spot_hex = (home_q, home_r)
        ant_hex = (ant.q, ant.r)

        # Если несёт еду - возвращаемся в колонию
        if ant.food:
            total_carried = ant.food
            if total_carried >= ant.capacity / 2 or ant_hex == spot_hex:
                if ant.hex == spot_hex:
                    return list()

                home_hex = choice(list(self.memory.home)).hex
                return self.find_path(ant.q, ant.r, *home_hex, state, subject=ant)

        # Угроза: рядом враг-солдат — убегаем к ближайшему своему солдату
        for enemy in state.enemies:
            if enemy.type == AntType.SOLDIER and PathFinder.hex_distance(ant_hex, enemy.hex) <= 5:
                friendly_soldiers = {a for a in state.ants if a.type == AntType.SOLDIER}
                if friendly_soldiers:
                    nearest_soldier = min(friendly_soldiers, key=lambda s: PathFinder.hex_distance(ant_hex, (s.q, s.r)))
                    return self.find_path(ant.q, ant.r, nearest_soldier.q, nearest_soldier.r, state, subject=ant)

        # Ищем ближайший запас еды, на который ещё не идут другие рабочие
        if state.food:
            for flag in range(2):
                available_food = {
                    f for f in state.food if f.amount > 0
                                             and state.get_hex(*f.hex).type != HexType.ANTHILL
                                             and ('ant', AntType.WORKER) not in state.who_at(state.get_hex(*f.hex))
                                             # and (flag ^ (state.get_hex(*f.hex).type == HexType.ACID))
                }

                if available_food:
                    target_food = min(available_food, key=lambda f: PathFinder.hex_distance(ant_hex, f.hex))
                    return self.find_path(ant.q, ant.r, target_food.q, target_food.r, state, subject=ant)

        # fallback — разведка
        return self.scout_strategy(ant, state)

    def find_path(self, start_q: int, start_r: int, target_q: int, target_r: int,
                  state: GameState, subject: Ant = None) -> List[Tuple[int, int]]:
        """
         Простой поиск пути с использованием алгоритма A*
         Возвращает путь от начальных координат до целевых
         """
        path = list()
        current = state.get_hex(start_q, start_r)
        finish = (target_q, target_r)
        opened = {(current, subject.speed, PathFinder.heuristic(current, finish), None)}  # Объект типа < Hex, max_length, Heuristic, previousHex >
        closed = dict()
        last = set()
        running = True

        while current.hex != finish and running:
            if not opened:
                break  # Нет пути

            current, speed, _, __ = min(opened, key=lambda h: PathFinder.hex_distance(h[0].hex, finish) + h[2])
            opened.remove((current, speed, _, __))
            closed[current] = (current, speed, _, __)
            # Все возможные клетки для передвижения
            neighbors = PathFinder.get_neighbors(*current.hex)

            # Удаляем невалидные гексы
            for nq, nr in neighbors:
                hex = state.get_hex(nq, nr)
                if hex and hex.type not in (HexType.STONE, ):
                    # Проверяем занят ли гекс
                    occupied = hex in closed
                    if subject is not None and not occupied:
                        for ant_team, ant_type in state.who_at(hex):
                            if subject.type == ant_type or ant_team == 'enemy':
                                occupied = True
                                break

                    n_speed = speed - hex.cost
                    if not occupied and n_speed >= 0:
                        if n_speed == 0:
                            running = False
                        last.add((state.get_hex(nq, nr), n_speed, 0, current))
                        opened.add((state.get_hex(nq, nr), n_speed, PathFinder.heuristic(hex, finish), current))

        # Выбираем ближайший подходящий гекс и строим путь
        if last:
            next_step = min(last, key=lambda h: PathFinder.hex_distance(h[0].hex, finish) + h[2])
        elif closed:
            next_step = min(closed.values(), key=lambda h: PathFinder.hex_distance(h[0].hex, finish) + h[2])
        else:
            next_step = min(opened, key=lambda h: PathFinder.hex_distance(h[0].hex, finish) + h[2])

        while next_step is not None:
            path.append(next_step[0].hex)
            next_step = closed.get(next_step[3])
        path.reverse()

        return path
