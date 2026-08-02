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

        self.known_map = {}
        if prev_map:
            for key, old_hex in prev_map.items():
                self.known_map[key] = Hex(
                    old_hex.q, old_hex.r, old_hex.type, old_hex.cost, is_visible=False
                )

        for hex_data in data.get('map', []):
            q, r = hex_data['q'], hex_data['r']
            h = Hex(q, r, HexType(hex_data['type']), hex_data['cost'], is_visible=True)
            self.known_map[(q, r)] = h

        for f in self.food:
            if (f.q, f.r) in self.known_map:
                self.known_map[(f.q, f.r)].food = f

        for ant in self.ants:
            if (ant.q, ant.r) in self.known_map:
                self.known_map[(ant.q, ant.r)].entity = ant

        for enemy in self.enemies:
            if (enemy.q, enemy.r) in self.known_map:
                self.known_map[(enemy.q, enemy.r)].entity = enemy

        self.map = list(self.known_map.values())

        self.spot = data.get('spot', {})
        self.next_turn_in = data.get('nextTurnIn', 0)
        self.score = data.get('score', 0)
        self.turn_no = data.get('turnNo', 0)

        self.borders = data.get('borders', set())

    def get_ant_by_id(self, ant_id: str) -> Optional[Ant]:
        for ant in self.ants:
            if ant.id == ant_id:
                return ant
        return None

    def get_hex(self, q: int, r: int) -> Optional[Hex]:
        return self.known_map.get((q, r))

    def get_food_at(self, q: int, r: int) -> Optional[Food]:
        for food in self.food:
            if food.q == q and food.r == r:
                return food
        return None

    def get_enemies_at(self, q: int, r: int) -> List[EnemyAnt]:
        return [enemy for enemy in self.enemies if enemy.q == q and enemy.r == r]

    def is_border(self, q: int, r: int) -> bool:
        return (q, r) in self.borders
