import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.server.server_engine import FastServerEngine, hex_distance

HEX_DIRECTIONS = [
    (0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
]

MAX_ANTS = 64
MAX_FOOD = 64
MAX_ENEMIES = 64

LIMITS = 512
TURNS = 300

HEX_EXPLORE_WEIGHT = 0.1
DAMAGE_WEIGHT = 0.3
ANTHILL_NEAR_WEIGHT = 0.3
SCORE_WEIGHT = 64.0
ANT_WEIGHT = 8.0
FOOD_WEIGHT = 4.0
FOOD_NEAR_WEIGHT = 0.05
MOVEMENT_WEIGHT = 0.01
HEAL_WEIGHT = 0.01


class FastAntWarsEnv(gym.Env):
    def __init__(self, num_bots: int = 2):
        super().__init__()
        self.num_bots = num_bots
        self.engine = FastServerEngine(num_teams=num_bots)
        self.agent_token = "rl_agent"
        self.bot_tokens = [f"bot_{i}" for i in range(1, num_bots)]
        self.prev_score = 0
        self.prev_ant_count = 0
        self.visited_hexes = set()

        self.action_space = spaces.MultiDiscrete([7] * MAX_ANTS)
        obs_dim = MAX_ANTS * 5 + MAX_FOOD * 3 + MAX_ENEMIES * 3
        self.observation_space = spaces.Box(low=-LIMITS, high=LIMITS, shape=(obs_dim,), dtype=np.float32)

    def get_action_masks(self) -> np.ndarray:
        """Генерирует маску допустимых действий размера (MAX_ANTS, 7)."""
        masks = np.zeros((MAX_ANTS, 7), dtype=bool)
        state = self.engine.get_state(self.agent_token)
        my_ants = state["ants"]

        for idx in range(MAX_ANTS):
            if idx >= len(my_ants):
                masks[idx, 0] = True  # Для несуществующих муравьёв разрешаем только покой
                continue

            ant = my_ants[idx]
            cq, cr = ant["q"], ant["r"]

            for act_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
                if dq == 0 and dr == 0:
                    masks[idx, act_idx] = True  # Стоять на месте можно всегда
                    continue

                tq, tr = cq + dq, cr + dr
                # Поддерживаем поиск по кортежу и по строке
                target_hex = self.engine.hex_map.get((tq, tr)) or self.engine.hex_map.get(f"{tq},{tr}")

                # ИСПРАВЛЕНО: Запрещаем движение ТОЛЬКО если гекс известен И это камень/стена
                if target_hex and (target_hex.get("type") == 5 or target_hex.get("cost", 1) >= 999):
                    continue

                # 2. Нельзя на чужой муравейник
                if any(t_tok != self.agent_token and any(h["q"] == tq and h["r"] == tr for h in t_data["home"])
                       for t_tok, t_data in self.engine.teams.items()):
                    continue

                # 3. Нельзя на гекс с врагом или сородичем того же типа
                blocker = next((a for a in self.engine.ants.values()
                                if a["health"] > 0 and a["q"] == tq and a["r"] == tr), None)
                if blocker and (blocker["team"] != self.agent_token or blocker["type"] == ant["type"]):
                    continue

                masks[idx, act_idx] = True

        return masks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        all_tokens = [self.agent_token] + self.bot_tokens
        self.engine.reset(all_tokens)

        self.prev_score = 0
        self.prev_ant_count = 1
        self.visited_hexes = set()
        state = self.engine.get_state(self.agent_token)
        return self._extract_obs(state), {"action_mask": self.get_action_masks()}

    def step(self, actions):
        state_before = self.engine.get_state(self.agent_token)
        my_ants = state_before["ants"]
        my_ants_before = {a["id"]: a for a in state_before["ants"]}

        # Формируем ходы агента
        agent_moves = []
        for idx, act in enumerate(actions):
            if idx < len(my_ants):
                ant = my_ants[idx]
                dq, dr = HEX_DIRECTIONS[act]
                if dq != 0 or dr != 0:
                    agent_moves.append({
                        "ant": ant["id"],
                        "path": [{"q": ant["q"] + dq, "r": ant["r"] + dr}]
                    })

        self.engine.pending_moves[self.agent_token] = agent_moves

        # Базовая эвристика для ботов-соперников
        for b_token in self.bot_tokens:
            b_state = self.engine.get_state(b_token)
            b_moves = []
            for b_ant in b_state["ants"]:
                if b_state["food"]:
                    target = min(b_state["food"], key=lambda f: hex_distance(b_ant["q"], b_ant["r"], f["q"], f["r"]))
                    dq = 1 if target["q"] > b_ant["q"] else (-1 if target["q"] < b_ant["q"] else 0)
                    dr = 1 if target["r"] > b_ant["r"] else (-1 if target["r"] < b_ant["r"] else 0)
                    if dq != 0 or dr != 0:
                        b_moves.append({"ant": b_ant["id"], "path": [{"q": b_ant["q"] + dq, "r": b_ant["r"] + dr}]})
            self.engine.pending_moves[b_token] = b_moves

        # Мгновенный просчёт хода
        self.engine.process_turn()

        state_after = self.engine.get_state(self.agent_token)
        obs = self._extract_obs(state_after)

        reward = self.calculate_reward(my_ants_before, state_after)
        done = self.engine.turn_no >= TURNS
        next_mask = self.get_action_masks()
        return obs, reward, done, False, {"action_mask": next_mask}

    def calculate_reward(self, my_ants_before, state_after) -> np.ndarray:
        rewards = np.zeros(MAX_ANTS, dtype=np.float32)
        score_diff = state_after["score"] - self.prev_score
        self.prev_score = state_after["score"]  # ИСПРАВЛЕНО: обновляем сохранённый счёт

        home_spot = state_after.get("spot", {"q": 0, "r": 0})

        for idx in range(MAX_ANTS):
            if idx >= len(state_after["ants"]):
                continue

            ant_after = state_after["ants"][idx]
            ant_id = ant_after["id"]
            ant_before = my_ants_before.get(ant_id)
            rewards[idx] += ANT_WEIGHT * (len(state_after["ants"]) - len(my_ants_before))

            if ant_before:
                pos_after = (ant_after["q"], ant_after["r"])

                # 0. Награда за исследование НОВОЙ клетки (+0.2)
                if pos_after not in self.visited_hexes:
                    rewards[idx] += HEX_EXPLORE_WEIGHT
                    self.visited_hexes.add(pos_after)

                # 1. Персональный урон
                hp_diff = ant_after["health"] - ant_before["health"]
                if hp_diff < 0:
                    rewards[idx] += hp_diff * DAMAGE_WEIGHT
                else:
                    rewards[idx] += hp_diff * HEAL_WEIGHT

                # 2. Персональный подбор еды
                if ant_before["food"] is None and ant_after["food"] is not None:
                    rewards[idx] += FOOD_WEIGHT

                # 3. Награда за сдачу ресурса
                if score_diff > 0 and ant_before["food"] is not None and ant_after["food"] is None:
                    rewards[idx] += score_diff * SCORE_WEIGHT

                # 4. Прогресс движения
                if ant_after["food"] is not None:
                    # Несёт еду -> к муравейнику
                    dist_before = hex_distance(ant_before["q"], ant_before["r"], home_spot["q"], home_spot["r"])
                    dist_after = hex_distance(ant_after["q"], ant_after["r"], home_spot["q"], home_spot["r"])
                    rewards[idx] += (dist_before - dist_after) * ANTHILL_NEAR_WEIGHT
                elif state_after["food"]:
                    # Ищет еду -> к ближайшей еде
                    closest_food = min(state_after["food"], key=lambda f: hex_distance(ant_before["q"], ant_before["r"], f["q"], f["r"]))
                    dist_before = hex_distance(ant_before["q"], ant_before["r"], closest_food["q"], closest_food["r"])
                    dist_after = hex_distance(ant_after["q"], ant_after["r"], closest_food["q"], closest_food["r"])
                    rewards[idx] += (dist_before - dist_after) * FOOD_NEAR_WEIGHT
                else:
                    # Маленький бонус за движение (чтобы не стоял на месте, когда еды не видно)
                    if ant_before["q"] != ant_after["q"] or ant_before["r"] != ant_after["r"]:
                        rewards[idx] += MOVEMENT_WEIGHT

                if ant_before["q"] != ant_after["q"] or ant_before["r"] != ant_after["r"]:
                    rewards[idx] += MOVEMENT_WEIGHT

        return rewards

    def _extract_obs(self, state: dict) -> np.ndarray:
        spot = state.get("spot", {"q": 0, "r": 0})
        sq, sr = spot["q"], spot["r"]
        obs = []

        # 1. Муравьи
        ants = state["ants"]
        for i in range(MAX_ANTS):
            if i < len(ants):
                a = ants[i]
                obs.extend([a["q"] - sq, a["r"] - sr, a["health"], a["type"], 1.0 if a["food"] else 0.0])
            else:
                obs.extend([0, 0, 0, -1, 0])

        # 2. Еда (берём первые N без медленной сортировки по всей карте)
        foods = state["food"][:MAX_FOOD]
        for i in range(MAX_FOOD):
            if i < len(foods):
                f = foods[i]
                obs.extend([f["q"] - sq, f["r"] - sr, f["amount"]])
            else:
                obs.extend([0, 0, 0])

        # 3. Враги
        enemies = state["enemies"][:MAX_ENEMIES]
        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                e = enemies[i]
                obs.extend([e["q"] - sq, e["r"] - sr, e["health"]])
            else:
                obs.extend([0, 0, 0])

        return np.array(obs, dtype=np.float32)
