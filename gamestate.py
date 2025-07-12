import time
from functools import lru_cache
import random
from threading import Thread

from models import *
from typing import *


class GameState:
    def __init__(self, data: Dict):
        self.memory = None
        self.ants = {Ant(ant['id'], AntType(ant['type']), ant['q'], ant['r'], ant['health'],
                         ant.get('food'), ant.get('lastMove'), ant.get('move'),
                         ant.get('lastAttack'), ant.get('lastEnemyAnt'))
                     for ant in data.get('ants', [])}

        self.enemies = {EnemyAnt(AntType(enemy['type']), enemy['q'], enemy['r'], enemy['health'],
                                 enemy.get('food'), enemy.get('attack'))
                        for enemy in data.get('enemies', [])}

        self.food = {Food(f['q'], f['r'], FoodType(f['type']), f['amount'])
                     for f in data.get('food', [])}

        self.home = [{'q': h['q'], 'r': h['r']} for h in data.get('home', [])]

        self.map = {Hex(hex['q'], hex['r'], HexType(hex['type']), hex['cost'])
                    for hex in data.get('map', [])}

        self.spot = data.get('spot', {})
        self.next_turn_in = data.get('nextTurnIn', 0)
        self.score = data.get('score', 0)
        self.turn_no = data.get('turnNo', 0)

    def get_ant_by_id(self, ant_id: str) -> Optional[Ant]:
        if self.memory is None:
            for ant in self.ants:
                if ant.id == ant_id:
                    return ant
            return None
        return self.memory.ants.get(ant_id, None)

    def who_at(self, hex: Hex) -> Set[Tuple[str, AntType]]:
        res = set()

        for ant in filter(lambda a: a.hex == hex.hex, self.ants | self.enemies):
                res.add(('ant' if isinstance(ant, Ant) else 'enemy', ant.type))

        return res

    @lru_cache(None)
    def get_hex(self, q: int, r: int) -> Optional[Hex]:
        for h in self.map:
            if h.q == q and h.r == r:
                return h
        return None

    @lru_cache(None)
    def get_hex_by_pair(self, hex: Tuple[int, int]) -> Optional[Hex]:
        if self.memory is None:
            return self.get_hex(*hex)
        return self.memory.map.get(hex, None)

    def get_food_at(self, q: int, r: int) -> Optional[Food]:
        if self.memory is None:
            for food in self.food:
                if food.q == q and food.r == r:
                    return food
            return None
        return self.memory.food.get((q, r), None)

    def get_enemies_at(self, q: int, r: int) -> List[EnemyAnt]:
        return {enemy for enemy in self.enemies if enemy.q == q and enemy.r == r}


class GameMemory:
    ants = dict()
    enemies = set()
    food = dict()
    home = set()
    spot: Hex = None
    enemy_hills = dict()
    map = dict()
    initialized = False
    have_attack = False
    enemy_hill = None
    roles = dict()
    thread: Thread

    def init_home(self, state: GameState):
        self.initialized = True
        self.home = set(map(lambda h: state.get_hex(h['q'], h['r']), state.home))
        self.spot = state.get_hex(*state.spot)

    def update(self, state: GameState):
        for ant in state.ants:
            self.ants[ant.id] = ant
        self.enemies = state.enemies
        for food in state.food:
            if food.amount > 0:
                self.food[food.hex] = food
        for ant in state.ants:
            self.ants[ant.id] = ant
        for hex in state.map:
            self.map[hex.hex] = hex
            if hex.type == HexType.ANTHILL and hex not in self.home:
                self.enemy_hills[hex.hex] = hex

    def start_attack(self, hill_cord):
        self.enemy_hill = self.enemy_hills.get(hill_cord, self.spot)
        if not self.have_attack:
            self.have_attack = True
            for ant in self.ants:
                if ant.type == AntType.WORKER:
                    if random.random() > 0.4:
                        self.roles[ant.id] = AntRole.HELPER
            self.thread = Thread(target=self._end_attack, daemon=True)

    def _end_attack(self):
        time.sleep(55)
        self.have_attack = False
        self.thread.join()
