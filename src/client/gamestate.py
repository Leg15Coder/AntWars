from src.client.models import *
from typing import *


class GameState:
    def __init__(self, data: Dict, prev_map: Optional[Dict[Tuple[int, int], Hex]] = None):
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

        self.known_map = dict(prev_map) if prev_map else {}

        for h in self.known_map.values():
            h.is_visible = False

        for hex_data in data.get('map', []):
            q, r = hex_data['q'], hex_data['r']
            h = Hex(q, r, HexType(hex_data['type']), hex_data['cost'], is_visible=True)
            self.known_map[(q, r)] = h

        self.map = list(self.known_map.values())

        self.spot = data.get('spot', {})
        self.next_turn_in = data.get('nextTurnIn', 0)
        self.score = data.get('score', 0)
        self.turn_no = data.get('turnNo', 0)

    def get_ant_by_id(self, ant_id: str) -> Optional[Ant]:
        for ant in self.ants:
            if ant.id == ant_id:
                return ant
        return None

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
