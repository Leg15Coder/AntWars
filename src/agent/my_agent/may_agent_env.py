import uuid
from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym

from src.server.server_engine import FastServerEngine, hex_distance
from src.client.models import AntType, Ant, HexType, EnemyAnt
from src.client.gamestate import GameState
from hex_pathfinding import astar_hex_path


MAX_OP = {
    AntType.SCOUT: (7, 4),    # (Очки Передвижения, Радиус Обзора)
    AntType.WORKER: (5, 1),
    AntType.SOLDIER: (4, 1)
}

HEX_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def axial_hex_distance(dq: int, dr: int) -> int:
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


MOVE_DIRECTIONS: Dict[AntType, List[Tuple[int, int]]] = {}
VISION_DIRECTIONS: Dict[AntType, List[Tuple[int, int]]] = {}
ACTION_DIMS: Dict[AntType, int] = {}
OBS_DIM: Dict[AntType, int] = {}

COLONY_SIGNAL_DIM = 12

for typ, (move_r, vis_r) in MAX_OP.items():
    MOVE_DIRECTIONS[typ] = [
        (q, r) for q in range(-move_r, move_r + 1)
        for r in range(-move_r, move_r + 1)
        if axial_hex_distance(q, r) <= move_r
    ]
    VISION_DIRECTIONS[typ] = [
        (q, r) for q in range(-vis_r, vis_r + 1)
        for r in range(-vis_r, vis_r + 1)
        if axial_hex_distance(q, r) <= vis_r
    ]
    ACTION_DIMS[typ] = len(MOVE_DIRECTIONS[typ])
    OBS_DIM[typ] = 4 + len(VISION_DIRECTIONS[typ]) * 5 + COLONY_SIGNAL_DIM


TURNS = 300

REWARD_WEIGHTS = {
    AntType.SCOUT: {
        "explore": 0.05, "damage_taken": 0.05, "damage_dealt": 0.005,
        "anthill_near": 0.001, "score": 2.0, "food_pick": 0.05,
        "food_near": 0.001, "movement": 0.005, "heal": 0.001,
    },
    AntType.WORKER: {
        "explore": 0.001, "damage_taken": 0.04, "damage_dealt": 0.005,
        "anthill_near": 0.2, "score": 8.0, "food_pick": 1.0,
        "food_near": 0.2, "movement": 0.001, "heal": 0.001,
    },
    AntType.SOLDIER: {
        "explore": 0.001, "damage_taken": 0.01, "damage_dealt": 0.5,
        "anthill_near": 0.01, "score": 4.0, "food_pick": 0.001,
        "food_near": 0.001, "movement": 0.002, "heal": 0.001,
    }
}


class AntWarsEnv(gym.Env):
    def __init__(self, num_bots: int = 2):
        super().__init__()
        self.num_bots = num_bots
        self.engine = FastServerEngine(num_teams=num_bots)
        self.agent_token = f"rl_agent_{uuid.uuid4()}"
        self.bot_tokens = [f"bot_{i}" for i in range(1, num_bots)]
        self.prev_score = 0
        self.visited_hexes = set()
        self.state: Optional[GameState] = None
        self.colony_signal = np.zeros(COLONY_SIGNAL_DIM, dtype=np.float32)

    def set_colony_signal(self, signal: np.ndarray):
        if len(signal) == COLONY_SIGNAL_DIM:
            self.colony_signal = np.array(signal, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        all_tokens = [self.agent_token] + self.bot_tokens
        self.engine.reset(all_tokens)

        self.prev_score = 0
        self.visited_hexes = set()
        self.colony_signal = np.zeros(COLONY_SIGNAL_DIM, dtype=np.float32)

        raw_state = self.engine.get_state(self.agent_token)
        self.state = GameState(raw_state)

        obs_by_type, masks_by_type, ants_by_type = self._get_env_state_by_type()

        info = {
            "masks_by_type": masks_by_type,
            "ants_by_type": ants_by_type
        }
        return obs_by_type, info

    def _get_env_state_by_type(self):
        obs_by_type : Dict[AntType, List[np.ndarray]] = {t: [] for t in AntType}
        ants_by_type : Dict[AntType, List[Ant]] = {t: [] for t in AntType}
        masks_by_type = {}

        for ant in self.state.ants:
            typ = ant.type
            obs_by_type[typ].append(self._extract_obs(ant))
            ants_by_type[typ].append(ant)

        for typ in AntType:
            masks_by_type[typ] = self.get_action_masks(ants_by_type[typ])
            if obs_by_type[typ]:
                obs_by_type[typ] = np.array(obs_by_type[typ], dtype=np.float32)
            else:
                obs_by_type[typ] = np.empty((0, OBS_DIM[typ]), dtype=np.float32)

        return obs_by_type, masks_by_type, ants_by_type

    def get_action_masks(self, ants: List[Ant]) -> np.ndarray:
        if not ants:
            return np.empty((0, 0), dtype=bool)

        typ = ants[0].type
        num_actions = ACTION_DIMS[typ]
        masks = np.zeros((len(ants), num_actions), dtype=bool)

        occupied_my = {(a.q, a.r): a for a in self.state.ants}
        occupied_enemies = {(e.q, e.r): e for e in getattr(self.state, "enemies", [])}

        for i, ant in enumerate(ants):
            ant_q, ant_r = ant.q, ant.r

            for j, (dq, dr) in enumerate(MOVE_DIRECTIONS[typ]):
                if dq == 0 and dr == 0:
                    masks[i, j] = True
                    continue

                tq, tr = ant_q + dq, ant_r + dr

                if self.state.is_border(tq, tr):
                    continue

                th = self.state.get_hex(tq, tr)
                # Разрешаем шаг в неизведанные клетки (th is None)
                if th is not None and th.type == HexType.STONE:
                    continue

                # Нельзя заходить на гекс с врагом
                if (tq, tr) in occupied_enemies:
                    continue

                # Нельзя заходить на гекс со своим муравьём того же типа
                blocker = occupied_my.get((tq, tr))
                if blocker and blocker.id != ant.id and blocker.type == ant.type:
                    continue

                masks[i, j] = True

        return masks

    def _extract_obs(self, ant: Ant) -> np.ndarray:
        spot = self.state.spot if self.state.spot else {"q": 0, "r": 0}  #[cite: 1]
        sq = spot.get("q", 0) if isinstance(spot, dict) else getattr(spot, "q", 0)
        sr = spot.get("r", 0) if isinstance(spot, dict) else getattr(spot, "r", 0)

        typ = ant.type
        ant_q, ant_r = ant.q, ant.r
        has_food_flag = 1.0 if ant.food is not None else 0.0  #

        obs = [ant_q - sq, ant_r - sr, float(ant.health), has_food_flag]

        for dq, dr in VISION_DIRECTIONS[typ]:
            tq, tr = ant_q + dq, ant_r + dr
            th = self.state.get_hex(tq, tr)

            if th is None:
                obs.extend([-1.0, 0.0, -1.0, -1.0, 0.0])
                continue

            # Еда
            if th.food is not None:
                f_type = th.food.type.value if hasattr(th.food.type, "value") else th.food.type
                obs.extend([float(f_type), float(th.food.amount)])
            else:
                obs.extend([-1.0, 0.0])

            # Тип клетки
            h_type = th.type.value if hasattr(th.type, "value") else th.type
            obs.append(float(h_type))

            # Сущность на клетке
            if th.entity is not None:
                is_enemy = 1.0 if isinstance(th.entity, EnemyAnt) else 0.0  #
                hp = float(getattr(th.entity, "health", 0))  #
                obs.extend([is_enemy, hp])
            else:
                obs.extend([-1.0, 0.0])

        obs.extend(self.colony_signal.tolist())
        return np.array(obs, dtype=np.float32)

    def step(self, actions_by_id: Dict[str, int], colony_signal: Optional[np.ndarray] = None):
        if colony_signal is not None:
            self.set_colony_signal(colony_signal)

        state_before = self.state
        my_ants_before = {a.id: a for a in state_before.ants}
        # Фиксируем врагов ДО проведения хода
        enemies_before = {(e.q, e.r): e.health for e in getattr(state_before, "enemies", [])}

        invalid_action_penalties = {}
        agent_moves = []

        for ant_id, act_idx in actions_by_id.items():
            ant = my_ants_before.get(ant_id)
            if ant is None:
                continue

            dirs = MOVE_DIRECTIONS[ant.type]
            if 0 <= act_idx < len(dirs):
                dq, dr = dirs[act_idx]
                if dq == 0 and dr == 0:
                    continue

                target_hex = (ant.q + dq, ant.r + dr)
                max_op = MAX_OP[ant.type][0]

                full_path = astar_hex_path(
                    start=(ant.q, ant.r),
                    target=target_hex,
                    max_op=max_op,
                    game_state=self.state
                )

                if not full_path:
                    best_dir = min(
                        HEX_NEIGHBORS,
                        key=lambda d: axial_hex_distance(dq - d[0], dr - d[1])
                    )
                    step_q, step_r = ant.q + best_dir[0], ant.r + best_dir[1]

                    full_path = astar_hex_path(
                        start=(ant.q, ant.r),
                        target=(step_q, step_r),
                        max_op=1,
                        game_state=self.state
                    )
                    invalid_action_penalties[ant.id] = -0.05

                if full_path:
                    agent_moves.append({
                        "ant": ant.id,
                        "path": full_path
                    })

        self.engine.pending_moves[self.agent_token] = agent_moves

        for b_token in self.bot_tokens:
            b_state = self.engine.get_state(b_token)
            b_moves = []
            for b_ant in b_state.get("ants", []):
                if b_state.get("food"):
                    target = min(
                        b_state["food"],
                        key=lambda f: hex_distance(b_ant["q"], b_ant["r"], f["q"], f["r"])
                    )
                    bot_path = astar_hex_path(
                        start=(b_ant["q"], b_ant["r"]),
                        target=(target["q"], target["r"]),
                        max_op=5,
                        game_state=self.state
                    )
                    if bot_path:
                        b_moves.append({"ant": b_ant["id"], "path": bot_path})
            self.engine.pending_moves[b_token] = b_moves

        self.engine.process_turn()

        raw_state_after = self.engine.get_state(self.agent_token)
        self.state = GameState(raw_state_after, prev_map=self.state.known_map)

        # Передаём сохранённый enemies_before
        rewards_by_id = self._calculate_rewards(my_ants_before, enemies_before, self.state)
        for ant_id, pen in invalid_action_penalties.items():
            if ant_id in rewards_by_id:
                rewards_by_id[ant_id] += pen

        done = self.engine.turn_no >= TURNS
        obs_by_type, masks_by_type, ants_by_type = self._get_env_state_by_type()

        info = {
            "masks_by_type": masks_by_type,
            "ants_by_type": ants_by_type,
            "rewards_by_id": rewards_by_id
        }

        return obs_by_type, rewards_by_id, done, False, info

    def _calculate_rewards(self, ants_before: Dict[str, Ant], enemies_before: Dict[Tuple[int, int], float], state_after: GameState) -> Dict[str, float]:
        rewards = {}
        spot = state_after.spot if state_after.spot else {"q": 0, "r": 0}
        sq = spot.get("q", 0) if isinstance(spot, dict) else getattr(spot, "q", 0)
        sr = spot.get("r", 0) if isinstance(spot, dict) else getattr(spot, "r", 0)

        ants_after = {a.id: a for a in state_after.ants}
        score_diff = state_after.score - self.prev_score
        self.prev_score = state_after.score

        enemies_after = {(e.q, e.r): e.health for e in getattr(state_after, "enemies", [])}

        for ant_id, ant_b in ants_before.items():
            ant_a = ants_after.get(ant_id)
            w = REWARD_WEIGHTS[ant_b.type]
            r = 0.0

            if ant_a is None:
                rewards[ant_id] = -10.0
                continue

            ant_a_hex = (ant_a.q, ant_a.r)
            ant_b_hex = (ant_b.q, ant_b.r)

            if ant_a_hex not in self.visited_hexes:
                r += w["explore"]
                self.visited_hexes.add(ant_a_hex)

            hp_diff = ant_a.health - ant_b.health
            if hp_diff < 0:
                r += hp_diff * w["damage_taken"]
            elif hp_diff > 0:
                r += hp_diff * w["heal"]

            for (eq, er), hp_b in enemies_before.items():
                if hex_distance(ant_b.q, ant_b.r, eq, er) <= 1:
                    hp_a = enemies_after.get((eq, er), 0.0)
                    damage_dealt = max(0.0, hp_b - hp_a)
                    r += damage_dealt * w["damage_dealt"]

            if ant_b.food is None and ant_a.food is not None:
                r += w["food_pick"]

            if score_diff > 0 and ant_b.food is not None and ant_a.food is None:
                r += score_diff * w["score"]

            if ant_a.food is not None:
                d_before = hex_distance(ant_b.q, ant_b.r, sq, sr)
                d_after = hex_distance(ant_a.q, ant_a.r, sq, sr)
                r += (d_before - d_after) * w["anthill_near"]
            elif getattr(state_after, "food", None):
                closest_f = min(
                    state_after.food,
                    key=lambda f: hex_distance(ant_b.q, ant_b.r, f.q, f.r)
                )
                d_before = hex_distance(ant_b.q, ant_b.r, closest_f.q, closest_f.r)
                d_after = hex_distance(ant_a.q, ant_a.r, closest_f.q, closest_f.r)
                r += (d_before - d_after) * w["food_near"]

            if ant_b_hex != ant_a_hex:
                r += w["movement"]

            rewards[ant_id] = r

        return rewards
