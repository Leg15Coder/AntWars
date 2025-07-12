from models import *
from typing import *


class GameState:
    def __init__(self, data: Dict):
        self.ants = [Ant(ant['id'], AntType(ant['type']), ant['q'], ant['r'], ant['health'],
                         ant.get('food'), ant.get('lastMove'), ant.get('move'),
                         ant.get('lastAttack'), ant.get('lastEnemyAnt'))
                     for ant in data.get('ants', [])]

        self.enemies = [EnemyAnt(AntType(enemy['type']), enemy['q'], enemy['r'], enemy['health'],
                                 enemy.get('food'), enemy.get('attack'))
                        for enemy in data.get('enemies', [])]

        self.food = [Food(f['q'], f['r'], FoodType(f['type']), f['amount'])
                     for f in data.get('food', [])]

        self.home = [{'q': h['q'], 'r': h['r']} for h in data.get('home', [])]

        self.map = [Hex(hex['q'], hex['r'], HexType(hex['type']), hex['cost'])
                    for hex in data.get('map', [])]

        self.spot = data.get('spot', {})
        self.next_turn_in = data.get('nextTurnIn', 0)
        self.score = data.get('score', 0)
        self.turn_no = data.get('turnNo', 0)

    def get_ant_by_id(self, ant_id: str) -> Optional[Ant]:
        for ant in self.ants:
            if ant.id == ant_id:
                return ant
        return None

    def who_at(self, hex: Hex) -> Set[Tuple[str, AntType]]:
        res = set()

        for ant in self.ants:
            if ant.hex == hex.hex:
                res.add(('ant', ant.type))

        for ant in self.enemies:
            if ant.hex == hex.hex:
                res.add(('enemy', ant.type))

        return res

    def get_hex(self, q: int, r: int) -> Optional[Hex]:
        for h in self.map:
            if h.q == q and h.r == r:
                return h
        return None

    def get_food_at(self, q: int, r: int) -> Optional[Food]:
        for food in self.food:
            if food.q == q and food.r == r:
                return food
        return None

    def get_enemies_at(self, q: int, r: int) -> List[EnemyAnt]:
        return [enemy for enemy in self.enemies if enemy.q == q and enemy.r == r]
