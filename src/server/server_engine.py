import math
import random
import time
from enum import Enum
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()
LOGS_PATH = os.getenv("LOGS_PATH")


# --- Настройки таймеров (в секундах) ---
LOBBY_DURATION = 15.0    # Длительность лобби для сбора игроков (по ТЗ 300.0 сек / 5 мин)
ROUND_DURATION = 600.0   # Длительность раунда (10 минут)
BREAK_DURATION = 60.0    # Технический перерыв (1 минута)
TURN_DURATION = 2.0      # Длительность одного хода (2 секунды)


class GamePhase(Enum):
    LOBBY = "LOBBY"
    RUNNING = "RUNNING"
    BREAK = "BREAK"


UNIT_PROPERTIES = {
    0: {"name": "WORKER", "hp": 130, "attack": 30, "capacity": 8, "view_range": 1, "speed": 5, "spawn_rate": 0.60},
    1: {"name": "SOLDIER", "hp": 180, "attack": 70, "capacity": 2, "view_range": 1, "speed": 4, "spawn_rate": 0.30},
    2: {"name": "SCOUT", "hp": 80, "attack": 20, "capacity": 2, "view_range": 4, "speed": 7, "spawn_rate": 0.10}
}

FOOD_CALORIES = {
    1: 10,  # яблоко
    2: 20,  # хлеб
    3: 60   # нектар
}


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2


def get_hex_neighbors(q: int, r: int) -> List[Tuple[int, int]]:
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    return [(q + dq, r + dr) for dq, dr in directions]


def get_hexes_in_radius(center_q: int, center_r: int, radius: int) -> List[Tuple[int, int]]:
    results = []
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
            results.append((center_q + dq, center_r + dr))
    return results


class FastServerEngine:
    def __init__(self, num_teams: int = 2, map_radius: int = 10):
        self.num_teams = num_teams
        self.map_radius = map_radius
        self.turn_no = 0
        self.teams: Dict[str, dict] = {}
        self.hex_map: Dict[Tuple[int, int], dict] = {}
        self.ants: Dict[str, dict] = {}
        self.food_spots: Dict[Tuple[int, int], dict] = {}
        self.pending_moves: Dict[str, List[dict]] = {}
        self.logs: Dict[str, List[str]] = {}

    def reset(self, team_tokens: List[str]):
        self.turn_no = 1
        self.teams.clear()
        self.hex_map.clear()
        self.ants.clear()
        self.food_spots.clear()
        self.pending_moves.clear()

        with open(rf'{LOGS_PATH}\server\log_at_{time.time()}.txt', 'w') as f:
            f.write(str(self.logs))
            self.logs.clear()

        # Генерация карты
        for q in range(-self.map_radius, self.map_radius + 1):
            for r in range(max(-self.map_radius, -q - self.map_radius), min(self.map_radius, -q + self.map_radius) + 1):
                hex_type, cost = 2, 1
                val = random.random()
                if val < 0.08:
                    hex_type, cost = 3, 2  # Грязь
                elif val < 0.12:
                    hex_type, cost = 4, 1  # Кислота
                elif val < 0.16 and (abs(q) > 2 or abs(r) > 2):
                    hex_type, cost = 5, 999  # Камень
                self.hex_map[(q, r)] = {"q": q, "r": r, "type": hex_type, "cost": cost}

        # Спавн муравейников
        dist = max(4, self.map_radius - 3)
        for idx, token in enumerate(team_tokens):
            angle = (2 * math.pi * idx) / len(team_tokens)
            cq = int(round(dist * math.cos(angle)))
            cr = int(round(dist * math.sin(angle)))
            spot = {"q": cq, "r": cr}
            neighbors = get_hex_neighbors(cq, cr)
            home_hexes = [spot, {"q": neighbors[0][0], "r": neighbors[0][1]}, {"q": neighbors[1][0], "r": neighbors[1][1]}]

            for h in home_hexes:
                self.hex_map[(h["q"], h["r"])] = {"q": h["q"], "r": h["r"], "type": 1, "cost": 1}

            self.teams[token] = {
                "token": token,
                "spot": spot,
                "home": home_hexes,
                "score": 0,
                "nectar": {f"{h['q']},{h['r']}": 0 for h in home_hexes}
            }
            self.spawn_ant(token, ant_type=0)

        for _ in range(len(team_tokens) * 3):
            self.spawn_food()

    def spawn_ant(self, token: str, ant_type: int) -> bool:
        team = self.teams.get(token)
        if not team:
            return False

        main_q, main_r = team["spot"]["q"], team["spot"]["r"]
        team_ants = [a for a in self.ants.values() if a["team"] == token and a["health"] > 0]

        if len(team_ants) >= 100:
            return False

        # Проверка лимита: если на основном гексе уже есть муравей того же типа
        if any(a for a in team_ants if a["q"] == main_q and a["r"] == main_r and a["type"] == ant_type):
            return False

        ant_id = f"ant_{token[:4]}_{len(self.ants) + 1}_{random.randint(10000, 99999)}"
        props = UNIT_PROPERTIES[ant_type]

        self.ants[ant_id] = {
            "id": ant_id,
            "team": token,
            "type": ant_type,
            "q": main_q,
            "r": main_r,
            "health": props["hp"],
            "max_health": props["hp"],
            "food": None,
            "lastMove": [],
            "move": [],
            "lastAttack": None,
            "lastEnemyAnt": None,
            "visited_this_turn": [(main_q, main_r)]
        }
        self.add_log(token, f"Муравей {ant_id} появился в муравейнике.")
        return True

    def spawn_food(self):
        empty = [pos for pos, h in self.hex_map.items() if h["type"] == 2 and pos not in self.food_spots]
        if empty:
            p = random.choice(empty)
            self.food_spots[p] = {"q": p[0], "r": p[1], "type": random.choice([1, 2, 3]), "amount": random.randint(10, 25)}

    def process_turn(self):
        """Фаза переходовки (каждые 2 секунды)."""
        self.turn_no += 1

        # 1. Атака муравейников
        for team in self.teams.values():
            for h in team["home"]:
                hq, hr = h["q"], h["r"]
                for ant in list(self.ants.values()):
                    if ant["team"] != team["token"] and ant["health"] > 0:
                        if hex_distance(hq, hr, ant["q"], ant["r"]) <= 2:
                            ant["health"] -= 20
                            self.add_log(ant["team"], f"Муравей {ant['id']} получил 20 урона от вражеского муравейника.")

        self.clean_dead_ants()

        # 2. Рандомизация порядка команд
        shuffled_teams = list(self.teams.keys())
        random.shuffle(shuffled_teams)

        # 3. Атака юнитов
        for token in shuffled_teams:
            team_ants = [a for a in self.ants.values() if a["team"] == token and a["health"] > 0]
            for ant in team_ants:
                if ant["health"] <= 0:
                    continue

                adj_enemies = [
                    e for e in self.ants.values()
                    if e["team"] != token and e["health"] > 0 and hex_distance(ant["q"], ant["r"], e["q"], e["r"]) == 1
                ]
                if adj_enemies:
                    target = random.choice(adj_enemies)
                    base_dmg = UNIT_PROPERTIES[ant["type"]]["attack"]

                    bonus = 0.0
                    # Бонус поддержки
                    if any(
                            other["team"] == token and other["id"] != ant["id"] and
                            hex_distance(other["q"], other["r"], ant["q"], ant["r"]) == 1 and
                            hex_distance(other["q"], other["r"], target["q"], target["r"]) == 1
                            for other in self.ants.values() if other["health"] > 0
                    ):
                        bonus += 0.50

                    # Бонус своего муравейника
                    spot = self.teams[token]["spot"]
                    if hex_distance(ant["q"], ant["r"], spot["q"], spot["r"]) <= 2:
                        bonus += 0.25

                    final_dmg = int(base_dmg * (1.0 + bonus))
                    target["health"] -= final_dmg
                    ant["lastAttack"] = {"q": target["q"], "r": target["r"]}
                    ant["lastEnemyAnt"] = target["id"]

        self.clean_dead_ants()

        # 4. Движение, сбор и сдача ресурсов
        for token in shuffled_teams:
            moves_dict = {m["ant"]: m.get("path", []) for m in self.pending_moves.get(token, [])}
            team_ants = [a for a in self.ants.values() if a["team"] == token and a["health"] > 0]

            for ant in team_ants:
                ant["visited_this_turn"] = [(ant["q"], ant["r"])]
                path = moves_dict.get(ant["id"], [])
                max_speed = UNIT_PROPERTIES[ant["type"]]["speed"]

                cq, cr = ant["q"], ant["r"]
                spent_op = 0
                executed_path = []

                for step in path:
                    tq, tr = step["q"], step["r"]
                    if hex_distance(cq, cr, tq, tr) != 1:
                        break

                    target_hex = self.hex_map.get((tq, tr))
                    if not target_hex or target_hex["type"] == 5:
                        break

                    op_cost = target_hex["cost"]
                    if spent_op + op_cost > max_speed:
                        break

                    # Нельзя на чужой муравейник
                    if any(t_tok != token and any(h["q"] == tq and h["r"] == tr for h in t_data["home"]) for t_tok, t_data in self.teams.items()):
                        break

                    # Нельзя на гекс с врагом или дружественным муравьём того же типа
                    blocker = next((a for a in self.ants.values() if a["health"] > 0 and a["q"] == tq and a["r"] == tr), None)
                    if blocker and (blocker["team"] != token or blocker["type"] == ant["type"]):
                        break

                    spent_op += op_cost
                    cq, cr = tq, tr
                    executed_path.append({"q": cq, "r": cr})
                    ant["visited_this_turn"].append((cq, cr))

                ant["q"], ant["r"] = cq, cr
                ant["lastMove"] = executed_path

                # Урон от кислоты
                if self.hex_map.get((cq, cr), {}).get("type") == 4:
                    ant["health"] -= 20

                # Сдача ресурсов
                is_home = any(h["q"] == cq and h["r"] == cr for h in self.teams[token]["home"])
                if is_home and ant["food"]:
                    cals = FOOD_CALORIES.get(ant["food"]["type"], 10) * ant["food"]["amount"]
                    self.teams[token]["score"] += cals
                    hk = f"{cq},{cr}"
                    self.teams[token]["nectar"][hk] = self.teams[token]["nectar"].get(hk, 0) + ant["food"]["amount"]
                    ant["food"] = None
                    self.add_log(token, f"Муравей {ant['id']} сдал ресурсы! Очки: +{cals}.")

                # Сбор ресурсов
                pos = (cq, cr)
                if not is_home and pos in self.food_spots:
                    f_info = self.food_spots[pos]
                    cap = UNIT_PROPERTIES[ant["type"]]["capacity"]
                    if ant["food"] is None:
                        take = min(cap, f_info["amount"])
                        ant["food"] = {"type": f_info["type"], "amount": take}
                        f_info["amount"] -= take
                        if f_info["amount"] <= 0:
                            del self.food_spots[pos]

        self.clean_dead_ants()
        self.pending_moves.clear()

        # 5. Спавн новых юнитов
        for token in self.teams.keys():
            rval = random.random()
            stype = 0 if rval < 0.60 else (1 if rval < 0.90 else 2)
            self.spawn_ant(token, stype)

        # 6. Спавн новых ресурсов
        if random.random() < 0.40:
            self.spawn_random_food()

    def spawn_random_food(self):
        empty = [
            pos for pos, h in self.hex_map.items()
            if h["type"] == 2 and pos not in self.food_spots and not any(a["q"] == pos[0] and a["r"] == pos[1] for a in self.ants.values())
        ]
        if empty:
            p = random.choice(empty)
            self.food_spots[p] = {"q": p[0], "r": p[1], "type": random.choice([1, 2, 3]), "amount": random.randint(10, 30)}

    def clean_dead_ants(self):
        dead_ids = [aid for aid, ant in self.ants.items() if ant["health"] <= 0]
        for aid in dead_ids:
            ant = self.ants[aid]
            if ant["food"]:
                pos = (ant["q"], ant["r"])
                if pos not in self.food_spots:
                    self.food_spots[pos] = {"q": ant["q"], "r": ant["r"], "type": ant["food"]["type"], "amount": ant["food"]["amount"]}
            del self.ants[aid]

    def add_log(self, token: str, msg: str):
        if token in self.logs:
            self.logs[token].append(f"[Ход #{self.turn_no}] {msg}")

    def get_state(self, token: str) -> dict:
        team = self.teams[token]
        visible_coords = set()
        visible_coords.update(get_hexes_in_radius(team["spot"]["q"], team["spot"]["r"], 2))

        for ant in self.ants.values():
            if ant["team"] == token and ant["health"] > 0:
                v_range = UNIT_PROPERTIES[ant["type"]]["view_range"]
                for vq, vr in ant.get("visited", [(ant["q"], ant["r"])]):
                    visible_coords.update(get_hexes_in_radius(vq, vr, v_range))

        my_ants = [a for a in self.ants.values() if a["team"] == token and a["health"] > 0]
        enemies = [a for a in self.ants.values() if a["team"] != token and a["health"] > 0 and (a["q"], a["r"]) in visible_coords]
        food = [f for pos, f in self.food_spots.items() if pos in visible_coords]

        return {
            "ants": my_ants,
            "enemies": enemies,
            "food": food,
            "spot": team["spot"],
            "score": team["score"],
            "turnNo": self.turn_no
        }
