from typing import *

from numpy import random

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

    def scout_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        # Ищем ближайшее тёмное пятно
        unexplored = self.find_closest_unexplored(ant.q, ant.r, state)
        if unexplored:
            self.find_path(ant.q, ant.r, ant.speed, unexplored[0], unexplored[1], state)
        return list()

    def soldier_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        # Ищем врага
        if state.enemies:
            closest_enemy = min(state.enemies,
                                key=lambda e: PathFinder.hex_distance((ant.q, ant.r), (e.q, e.r)))
            if PathFinder.hex_distance((ant.q, ant.r), (closest_enemy.q, closest_enemy.r)) <= 1:
                # Если сражаемся - не надо двигаться
                return list()
            return self.find_path(ant.q, ant.r, ant.speed, closest_enemy.q, closest_enemy.r, state)

        # Если нет врагов - патрулируем колонию
        home_hex = state.spot
        if (ant.q, ant.r) == (home_hex['q'], home_hex['r']):
            neighbors = PathFinder.hex_in_area((ant.q, ant.r), big_radius=3)
            valid_neighbors = [n for n in neighbors if state.get_hex(n[0], n[1]) and
                               state.get_hex(n[0], n[1]).type not in (HexType.STONE, HexType.ANTHILL, HexType.ACID)]
            if valid_neighbors:
                return [random.choice(valid_neighbors)]
        else:
            return self.find_path(ant.q, ant.r, ant.speed, home_hex['q'], home_hex['r'], state)

        return list()

    def worker_strategy(self, ant: Ant, state: GameState) -> List[Tuple[int, int]]:
        # Если несёт еду - возвращаемся в колонию
        if ant.food:
            home_hex = state.spot
            if ant.hex == (home_hex['q'], home_hex['r']):
                return list()  # уже в колонии
            return self.find_path(ant.q, ant.r, ant.speed, home_hex['q'], home_hex['r'], state)

        # Видим еду - идём за ней
        if state.food:
            closest_food = min(filter(lambda food: PathFinder.hex_distance(ant.hex, food.hex) < 7, state.food),
                               key=lambda f: PathFinder.hex_distance((ant.q, ant.r), (f.q, f.r)))
            if (ant.q, ant.r) == (closest_food.q, closest_food.r):
                return list()  # уже в колонии
            return self.find_path(ant.q, ant.r, ant.speed, closest_food.q, closest_food.r, state)

        # Если нет еды - исследуем территорию в её поисках
        return self.scout_strategy(ant, state)

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
                        if ant.q == nq and ant.r == nr and ant.id != state.get_ant_by_id(current_q, current_r):
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
