from typing import *
from enum import Enum


class HexType(Enum):
    ANTHILL = 1
    EMPTY = 2
    DIRT = 3
    ACID = 4
    STONE = 5


class AntType(Enum):
    WORKER = 0
    SOLDIER = 1
    SCOUT = 2


class FoodType(Enum):
    APPLE = 1
    BREAD = 2
    NECTAR = 3


ANTS_PROPERTY = {
    AntType.SCOUT: {
        'max_health': 80,
        'damage': 20,
        'capacity': 2,
        'view_range': 4,
        'speed': 7,
        'spawn_rate': 0.1
    },
    AntType.SOLDIER: {
            'max_health': 180,
            'damage': 70,
            'capacity': 2,
            'view_range': 1,
            'speed': 4,
            'spawn_rate': 0.3
        },
    AntType.WORKER: {
            'max_health': 130,
            'damage': 30,
            'capacity': 8,
            'view_range': 1,
            'speed': 5,
            'spawn_rate': 0.6
        },
}


class Hex:
    def __init__(self, q: int, r: int, hex_type: HexType, cost: int):
        self.q = q
        self.r = r
        self.hex = (self.q, self.r)
        self.type = hex_type
        self.cost = cost

    def __str__(self):
        return f"Hex({self.q}, {self.r}, {self.type.name}, cost={self.cost})"

    def __eq__(self, other):
        if isinstance(other, Hex) and self.hex == other.hex:
            return True
        return False

    def __hash__(self):
        return hash(self.hex)


class Food:
    def __init__(self, q: int, r: int, food_type: FoodType, amount: int):
        self.q = q
        self.r = r
        self.hex = (self.q, self.r)
        self.type = food_type
        self.amount = amount

    def __str__(self):
        return f"Food({self.q}, {self.r}, {self.type.name}, amount={self.amount})"

    def __eq__(self, other):
        if isinstance(other, Food) and self.type == other.type and self.hex == other.hex:
            return True
        return False

    def __hash__(self):
        return hash((self.type, self.hex))


class Ant:
    def __init__(self, ant_id: str, ant_type: AntType, q: int, r: int, health: int,
                 food: Optional[Dict] = None, last_move: Optional[List[Dict]] = None,
                 move: Optional[List[Dict]] = None, last_attack: Optional[Dict] = None,
                 last_enemy_ant: Optional[str] = None):
        self.id = ant_id
        self.type = ant_type

        for attr, val in ANTS_PROPERTY.get(self.type, dict()).items():
            exec(f"self.{attr} = {val}")

        self.q = q
        self.r = r
        self.hex = (self.q, self.r)
        self.health = health
        self.food = food["amount"] if food and "amount" in food else None
        self.last_move = last_move
        self.move = move
        self.last_attack = last_attack
        self.last_enemy_ant = last_enemy_ant

    def __str__(self):
        return f"Ant({self.id}, {self.type.name}, pos=({self.q}, {self.r}), health={self.health})"

    def __eq__(self, other):
        if isinstance(other, Ant) and self.id == other.id:
            return True
        return False

    def __hash__(self):
        return hash(self.id)


class EnemyAnt:
    def __init__(self, ant_type: AntType, q: int, r: int, health: int,
                 food: Optional[Dict] = None, attack: Optional[int] = None):
        self.type = ant_type

        for attr, val in ANTS_PROPERTY.get(self.type, dict()).items():
            exec(f"self.{attr} = {val}")

        self.q = q
        self.r = r
        self.hex = (self.q, self.r)
        self.health = health
        self.food = food
        self.attack = attack

    def __str__(self):
        return f"EnemyAnt({self.type.name}, pos=({self.q}, {self.r}), health={self.health})"

    def __eq__(self, other):
        if isinstance(other, EnemyAnt) and self.type == other.type and self.hex == other.hex:
            return True
        return False

    def __hash__(self):
        return hash((self.type, self.hex))
