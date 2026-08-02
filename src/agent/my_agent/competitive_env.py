import uuid
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

from may_agent_env import AntWarsEnv, AntType, Ant, MAX_OP, HEX_NEIGHBORS, axial_hex_distance
from src.client.gamestate import GameState
from src.client.algorithm import Strategy
from hex_pathfinding import astar_hex_path
from src.agent.train import DQNAgent, MAX_ANTS
from src.agent.one_ant_agent.train_single_agent import SingleAntDQNAgent, extract_single_ant_obs, SINGLE_OBS_DIM


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Универсальное извлечение поля из объекта или словаря."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    if hasattr(val, "value"):
        return val.value
    return val


class MockClient:
    """Фейковый клиент для перехвата ходов эвристической Стратегии."""
    def __init__(self):
        self.last_moves = []

    def send_moves(self, moves: List[Dict]) -> bool:
        self.last_moves = moves
        return True


class CompetitiveAntWarsEnv(AntWarsEnv):
    """
    Мульти-ролевая среда AntWarsEnv, где соперниками выступают:
    - Bot 1: Эвристическая алгоритмическая стратегия (Strategy без ИИ)
    - Bot 2: Глобальный RL-агент (DQNAgent)
    - Bot 3: Локальный RL-агент одного муравья (SingleAntDQNAgent)
    """

    def __init__(
        self,
        dqn_model_path: Optional[str] = None,
        single_dqn_model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        super().__init__(num_bots=4)
        self.device = torch.device(device)

        # 1. Инициализация Bot 1 (Классический эвристический алгоритм)
        self.mock_client = MockClient()
        self.heuristic_strategy = Strategy(self.mock_client)

        # 2. Инициализация Bot 2 (Глобальный DQNAgent)
        obs_dim = MAX_ANTS * 5 + 64 * 3 + 64 * 3
        self.global_dqn = DQNAgent(obs_dim)
        self.global_dqn.epsilon = 0.05

        if dqn_model_path and Path(dqn_model_path).exists():
            self.global_dqn.model.load_state_dict(
                torch.load(dqn_model_path, map_location=self.device)
            )
            print(f"[CompetitiveEnv] Загружена глобальная модель DQN: {dqn_model_path}")
        self.global_dqn.model.eval()

        # 3. Инициализация Bot 3 (Single-Ant DQNAgent)
        self.single_dqn = SingleAntDQNAgent(SINGLE_OBS_DIM)
        self.single_dqn.epsilon = 0.05

        if single_dqn_model_path and Path(single_dqn_model_path).exists():
            self.single_dqn.model.load_state_dict(
                torch.load(single_dqn_model_path, map_location=self.device)
            )
            print(f"[CompetitiveEnv] Загружена модель SingleAntDQN: {single_dqn_model_path}")
        self.single_dqn.model.eval()

    def _extract_flat_obs_for_bot(self, state: dict) -> np.ndarray:
        """Безопасное формирование плоского вектора наблюдений для глобального DQN-бота."""
        spot = get_field(state, "spot") or {"q": 0, "r": 0}
        sq = get_field(spot, "q", 0)
        sr = get_field(spot, "r", 0)
        obs = []

        ants = get_field(state, "ants", []) or []
        for i in range(MAX_ANTS):
            if i < len(ants):
                a = ants[i]
                obs.extend([
                    get_field(a, "q", 0) - sq,
                    get_field(a, "r", 0) - sr,
                    get_field(a, "health", 0),
                    get_field(a, "type", 0),
                    1.0 if get_field(a, "food") is not None else 0.0
                ])
            else:
                obs.extend([0, 0, 0, -1, 0])

        foods = (get_field(state, "food", []) or [])[:64]
        for i in range(64):
            if i < len(foods):
                f = foods[i]
                obs.extend([
                    get_field(f, "q", 0) - sq,
                    get_field(f, "r", 0) - sr,
                    get_field(f, "amount", 0)
                ])
            else:
                obs.extend([0, 0, 0])

        enemies = (get_field(state, "enemies", []) or [])[:64]
        for i in range(64):
            if i < len(enemies):
                e = enemies[i]
                obs.extend([
                    get_field(e, "q", 0) - sq,
                    get_field(e, "r", 0) - sr,
                    get_field(e, "health", 0)
                ])
            else:
                obs.extend([0, 0, 0])

        return np.array(obs, dtype=np.float32)

    def get_bot_action_masks(self, bot_token: str) -> np.ndarray:
        """Генерирует маску допустимых ходов для бота размера (MAX_ANTS, 7)."""
        masks = np.zeros((MAX_ANTS, 7), dtype=bool)
        state = self.engine.get_state(bot_token)
        my_ants = get_field(state, "ants", []) or []

        HEX_DIRS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        for idx in range(MAX_ANTS):
            if idx >= len(my_ants):
                masks[idx, 0] = True
                continue

            ant = my_ants[idx]
            cq = get_field(ant, "q", 0)
            cr = get_field(ant, "r", 0)
            ant_type = get_field(ant, "type", 0)

            for act_idx, (dq, dr) in enumerate(HEX_DIRS):
                if dq == 0 and dr == 0:
                    masks[idx, act_idx] = True
                    continue

                tq, tr = cq + dq, cr + dr
                target_hex = self.state.get_hex(tq, tr)

                if target_hex and (get_field(target_hex, "type") == 5 or get_field(target_hex, "cost", 1) >= 999):
                    continue

                if any(t_tok != bot_token and any(get_field(h, "q") == tq and get_field(h, "r") == tr for h in get_field(t_data, "home", []))
                       for t_tok, t_data in self.engine.teams.items()):
                    continue

                blocker = next((a for a in self.engine.ants.values()
                                if a["health"] > 0 and a["q"] == tq and a["r"] == tr), None)
                if blocker and (blocker["team"] != bot_token or blocker["type"] == ant_type):
                    continue

                masks[idx, act_idx] = True

        return masks

    def step(self, actions_by_id: Dict[str, int], colony_signal: Optional[np.ndarray] = None):
        if colony_signal is not None:
            self.set_colony_signal(colony_signal)

        state_before = self.state
        my_ants_before = {a.id: a for a in state_before.ants}
        enemies_before = {(e.q, e.r): e.health for e in getattr(state_before, "enemies", [])}

        invalid_action_penalties = {}
        agent_moves = []

        # 1. Ходы главного RL-агента
        for ant_id, act_idx in actions_by_id.items():
            ant = my_ants_before.get(ant_id)
            if ant is None:
                continue

            HEX_DIRS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            if 0 <= act_idx < len(HEX_DIRS):
                dq, dr = HEX_DIRS[act_idx]
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

        # 2. Симуляция Bot 1: Эвристика
        if len(self.bot_tokens) >= 1:
            bot1_token = self.bot_tokens[0]
            b1_raw_state = self.engine.get_state(bot1_token)
            b1_game_state = GameState(b1_raw_state)

            self.mock_client.last_moves = []
            self.heuristic_strategy.make_turn(b1_game_state)
            self.engine.pending_moves[bot1_token] = self.mock_client.last_moves

        # 3. Симуляция Bot 2: Глобальный DQNAgent
        if len(self.bot_tokens) >= 2:
            bot2_token = self.bot_tokens[1]
            b2_state_dict = self.engine.get_state(bot2_token)
            b2_obs = self._extract_flat_obs_for_bot(b2_state_dict)
            b2_mask = self.get_bot_action_masks(bot2_token)

            b2_actions = self.global_dqn.act(b2_obs, b2_mask)
            HEX_DIRS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

            b2_moves = []
            b2_ants = get_field(b2_state_dict, "ants", []) or []
            for idx, act in enumerate(b2_actions):
                if idx < len(b2_ants):
                    ant = b2_ants[idx]
                    ant_id = get_field(ant, "id")
                    ant_q = get_field(ant, "q", 0)
                    ant_r = get_field(ant, "r", 0)

                    dq, dr = HEX_DIRS[act]
                    if dq != 0 or dr != 0:
                        b2_moves.append({
                            "ant": ant_id,
                            "path": [{"q": ant_q + dq, "r": ant_r + dr}]
                        })
            self.engine.pending_moves[bot2_token] = b2_moves

        # 4. Симуляция Bot 3: Single-Ant DQNAgent
        if len(self.bot_tokens) >= 3:
            bot3_token = self.bot_tokens[2]
            b3_state_dict = self.engine.get_state(bot3_token)
            b3_ants = get_field(b3_state_dict, "ants", []) or []
            b3_mask = self.get_bot_action_masks(bot3_token)

            HEX_DIRS = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            b3_moves = []
            for i, ant in enumerate(b3_ants[:MAX_ANTS]):
                ant_obs = extract_single_ant_obs(b3_state_dict, ant)
                ant_mask = b3_mask[i]

                act_i = self.single_dqn.act(ant_obs, ant_mask)
                dq, dr = HEX_DIRS[act_i]
                if dq != 0 or dr != 0:
                    ant_id = get_field(ant, "id")
                    ant_q = get_field(ant, "q", 0)
                    ant_r = get_field(ant, "r", 0)

                    b3_moves.append({
                        "ant": ant_id,
                        "path": [{"q": ant_q + dq, "r": ant_r + dr}]
                    })
            self.engine.pending_moves[bot3_token] = b3_moves

        # 5. Движок обрабатывает ход
        self.engine.process_turn()

        raw_state_after = self.engine.get_state(self.agent_token)
        self.state = GameState(raw_state_after, prev_map=self.state.known_map)

        rewards_by_id = self._calculate_rewards(my_ants_before, enemies_before, self.state)
        for ant_id, pen in invalid_action_penalties.items():
            if ant_id in rewards_by_id:
                rewards_by_id[ant_id] += pen

        done = self.engine.turn_no >= 300
        obs_by_type, masks_by_type, ants_by_type = self._get_env_state_by_type()

        info = {
            "masks_by_type": masks_by_type,
            "ants_by_type": ants_by_type,
            "rewards_by_id": rewards_by_id
        }

        return obs_by_type, rewards_by_id, done, False, info
