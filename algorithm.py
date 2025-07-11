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
        hexes = []

        for q in range(-big_radius, big_radius + 1):
            for r in range(max(-big_radius, -q - big_radius), min(big_radius, -q + big_radius) + 1):
                distance = PathFinder.hex_distance((0, 0), (q, r))

                if (distance > small_radius or (not is_small_strict and distance == small_radius)) and \
                        (distance < big_radius or (not is_big_strict and distance == big_radius)):
                    hex_q = center_q + q
                    hex_r = center_r + r
                    hexes.append((hex_q, hex_r))

        return hexes


class Strategy:
    def __init__(self, client: DatsPulseClient):
        self.client = client

    def make_turn(self):
        state = self.client.get_arena_state()

        moves = list()

        for ant in state.ants:
            if ant.type == AntType.SCOUT:
                path = self.scout_strategy(ant, state)
            elif ant.type == AntType.SOLDIER:
                path = self.soldier_strategy(ant, state)
            else:  # AntType.WORKER
                path = self.worker_strategy(ant, state)

            if path:
                moves.append({
                    'ant': ant.id,
                    'path': [{'q': q, 'r': r} for q, r in path]
                })

        if moves:
            self.client.send_moves(moves)

    def recall_all_soldiers_to_base(self, state: GameState):
        base_q, base_r = state.spot['q'], state.spot['r']
        moves = []

        for ant in state.ants:
            if ant.type == AntType.SOLDIER:
                path = self.find_path(ant.q, ant.r, ANTS_PROPERTY[ant.type]['speed'], base_q, base_r, state)
                if path:
                    moves.append({
                        'ant': ant.id,
                        'path': [{'q': q, 'r': r} for q, r in path]
                    })

        if moves:
            self.client.send_moves(moves)

    def recall_all_workers_to_base(self, state: GameState):
        base_q, base_r = state.spot['q'], state.spot['r']
        moves = []

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

    def scout_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        # Ищем ближайшее тёмное пятно
        speed = ANTS_PROPERTY[ant.type]['speed']
        unexplored = self.find_closest_unexplored(ant.q, ant.r, state)
        if unexplored:
            return self.find_path(ant.q, ant.r, speed, unexplored[0], unexplored[1], state)
        return list()

    def worker_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        home_q, home_r = state.spot['q'], state.spot['r']
        home_hex = (home_q, home_r)
        ant_hex = (ant.q, ant.r)
        speed = ANTS_PROPERTY[ant.type]['speed']
        capacity = ANTS_PROPERTY[ant.type]['capacity']

        # Если муравей несёт еду хотя бы на 50% — идём на базу
        if ant.food:
            total_carried = sum(ant.food.values()) if isinstance(ant.food, dict) else 0
            if total_carried >= capacity / 2 or ant_hex == home_hex:
                return self.find_path(ant.q, ant.r, speed, home_q, home_r, state)

        # Угроза: рядом враг-солдат — убегаем к ближайшему своему солдату
        for enemy in state.enemies:
            if enemy.type == AntType.SOLDIER and PathFinder.hex_distance(ant_hex, enemy.hex) <= 5:
                friendly_soldiers = [a for a in state.ants if a.type == AntType.SOLDIER]
                if friendly_soldiers:
                    nearest_soldier = min(friendly_soldiers, key=lambda s: PathFinder.hex_distance(ant_hex, (s.q, s.r)))
                    return self.find_path(ant.q, ant.r, speed, nearest_soldier.q, nearest_soldier.r, state)

        # Ищем ближайший запас еды, на который ещё не идут другие рабочие
        if state.food:
            targeted = set()
            for a in state.ants:
                if a.type == AntType.WORKER and a.id != ant.id and a.move and len(a.move) > 0:
                    last_step = a.move[-1]
                    targeted.add((last_step['q'], last_step['r']))

            available_food = [
                f for f in state.food if f.amount > 0 and (f.q, f.r) not in targeted
            ]

            if available_food:
                target_food = min(available_food, key=lambda f: PathFinder.hex_distance(ant_hex, f.hex))
                return self.find_path(ant.q, ant.r, speed, target_food.q, target_food.r, state)

        # fallback — разведка
        return self.scout_strategy(ant, state)

    def soldier_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        home_q, home_r = state.spot['q'], state.spot['r']
        home_hex = (home_q, home_r)
        ant_hex = (ant.q, ant.r)
        speed = ANTS_PROPERTY[ant.type]['speed']
        radius = 7
        spacing = 3

        # Кольцевые позиции на расстоянии 7 клеток
        ring_positions = PathFinder.hex_in_area(home_hex, small_radius=radius, big_radius=radius)
        ring_positions = sorted(ring_positions, key=lambda h: (h[0], h[1]))[::spacing]

        # Позиции, которые уже заняты
        reserved = set()
        for a in state.ants:
            if a.type == AntType.SOLDIER:
                if a.move and len(a.move) > 0:
                    reserved.add((a.move[-1]['q'], a.move[-1]['r']))
                elif PathFinder.hex_distance((a.q, a.r), home_hex) == radius:
                    reserved.add((a.q, a.r))

        # Свободные позиции
        free_positions = [pos for pos in ring_positions if pos not in reserved]

        if free_positions:
            nearest_free = min(free_positions, key=lambda pos: PathFinder.hex_distance(ant_hex, pos))
            return self.find_path(ant.q, ant.r, speed, nearest_free[0], nearest_free[1], state)

        # Сопровождаем рабочих, идущих от базы
        workers_on_mission = [
            a for a in state.ants
            if a.type == AntType.WORKER and a.food and PathFinder.hex_distance((a.q, a.r), home_hex) > 3
        ]
        if workers_on_mission:
            nearest_worker = min(workers_on_mission, key=lambda w: PathFinder.hex_distance(ant_hex, (w.q, w.r)))
            return self.find_path(ant.q, ant.r, speed, nearest_worker.q, nearest_worker.r, state)

        return []

    def find_path(self, start_q: int, start_r: int, speed: int, target_q: int, target_r: int,
                  state: GameState) -> List[Tuple[int, int]]:
        """
         Простой поиск пути с использованием алгоритма A*
         Возвращает путь от начальных координат до целевых
         """
        # TODO: реализовать правильный поиск пути с шестнадцатеричными затратами и препятствиями
        # Это упрощенная версия, которая просто движется прямо к цели
        path = list()
        current_q, current_r = start_q, start_r

        while (current_q, current_r) != (target_q, target_r):
            # Все возможные клетки для передвижения
            neighbors = PathFinder.get_neighbors(current_q, current_r)

            # Удаляем невалидные гексы
            valid_neighbors = list()
            for nq, nr in neighbors:
                hex = state.get_hex(nq, nr)
                if hex and hex.type not in (HexType.STONE, HexType.ANTHILL):
                    # Проверяем занят ли гекс
                    occupied = False
                    for ant in state.ants:
                        if ant.q == nq and ant.r == nr: #and ant.id != state.get_ant_by_id(current_q, current_r):
                            occupied = True
                            break
                    if not occupied:
                        valid_neighbors.append((nq, nr))

            if not valid_neighbors:
                break  # Нет пути

            # Выбираем ближайший подходящий гекс
            next_step = min(valid_neighbors,
                            key=lambda h: PathFinder.hex_distance(h, (target_q, target_r)))
            path.append(next_step)
            current_q, current_r = next_step
            speed -= 1
            if speed == 0:
                break

        return path

    def find_closest_unexplored(self, q: int, r: int, state: GameState) -> Optional[Tuple[int, int]]:
        edge_hexes = list()
        for hex in state.map:
            neighbors = PathFinder.get_neighbors(hex.q, hex.r)
            for nq, nr in neighbors:
                if not state.get_hex(nq, nr):
                    edge_hexes.append((nq, nr))

        if edge_hexes:
            return min(edge_hexes, key=lambda h: PathFinder.hex_distance((q, r), h))
        return None
