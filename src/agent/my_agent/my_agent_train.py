import os
import random
import time
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Импортируем нашу среду и ключевые типы
from may_agent_env import (
    AntWarsEnv,
    AntType,
    OBS_DIM,
    ACTION_DIMS,
    COLONY_SIGNAL_DIM,
    TURNS
)
from competitive_env import CompetitiveAntWarsEnv


# Масштабирующий коэффициент для стабилизации наград
REWARD_SCALE = 0.01


# -------------------------------------------------------------
# 1. Нейросеть Коллективного Разума (Глобальный Мозг Колонии)
# -------------------------------------------------------------
class ColonyBrainNetwork(nn.Module):
    def __init__(self, global_input_dim: int = 16, signal_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, signal_dim),
            nn.Tanh()  # Сигнал в диапазоне [-1, 1]
        )
        # Обучаемый параметр дисперсии для исследования
        self.log_std = nn.Parameter(torch.zeros(signal_dim) - 0.5)

    def forward(self, x: torch.Tensor):
        mu = self.net(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        raw_signal = dist.rsample()
        signal = torch.clamp(raw_signal, -1.0, 1.0)
        log_prob = dist.log_prob(raw_signal).sum(dim=-1)

        return signal, log_prob


def extract_global_features(env: AntWarsEnv) -> np.ndarray:
    """Собирает сжатую векторную сводку всей карты и ресурсов для Глобального мозга."""
    state = env.state
    if state is None:
        return np.zeros(16, dtype=np.float32)

    spot = state.spot if state.spot else {"q": 0, "r": 0}
    sq = spot.get("q", 0) if isinstance(spot, dict) else getattr(spot, "q", 0)
    sr = spot.get("r", 0) if isinstance(spot, dict) else getattr(spot, "r", 0)

    num_scouts = sum(1 for a in state.ants if a.type == AntType.SCOUT)
    num_workers = sum(1 for a in state.ants if a.type == AntType.WORKER)
    num_soldiers = sum(1 for a in state.ants if a.type == AntType.SOLDIER)
    num_enemies = len(getattr(state, "enemies", []))
    num_food = len(getattr(state, "food", []))

    features = [
        float(sq), float(sr),
        float(state.score),
        float(len(state.ants)),
        float(num_scouts), float(num_workers), float(num_soldiers),
        float(num_enemies), float(num_food),
        float(env.engine.turn_no),
        float(len(env.visited_hexes)),
        0.0, 0.0, 0.0, 0.0, 0.0  # Резервный паддинг до 16 фичей
    ]
    return np.array(features[:16], dtype=np.float32)


# -------------------------------------------------------------
# 2. Ролевая Q-Сеть (для конкретного типа муравья)
# -------------------------------------------------------------
class RoleQNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------
# 3. Мульти-ролевой управляющий агент
# -------------------------------------------------------------
class MultiRoleDQNAgent:
    def __init__(self, device: torch.device):
        self.device = device

        self.models: Dict[AntType, RoleQNetwork] = {}
        self.targets: Dict[AntType, RoleQNetwork] = {}
        self.optimizers: Dict[AntType, optim.Adam] = {}
        self.memories: Dict[AntType, deque] = {}

        for typ in AntType:
            in_dim = OBS_DIM[typ]
            act_dim = ACTION_DIMS[typ]

            model = RoleQNetwork(in_dim, act_dim).to(self.device)
            target = RoleQNetwork(in_dim, act_dim).to(self.device)
            target.load_state_dict(model.state_dict())

            self.models[typ] = model
            self.targets[typ] = target
            self.optimizers[typ] = optim.Adam(model.parameters(), lr=3e-4)
            self.memories[typ] = deque(maxlen=100000)

        # Модель Коллективного разума
        self.colony_brain = ColonyBrainNetwork(global_input_dim=16).to(self.device)
        self.colony_optimizer = optim.Adam(self.colony_brain.parameters(), lr=1e-4)
        self.last_colony_log_prob = None
        self.colony_reward_history = deque(maxlen=32)  # Baseline для стабилизации REINFORCE

        # Гиперпараметры RL
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.09  # Защита от полного вырождения исследований
        self.epsilon_decay = 0.997
        self.batch_size = 64

    def get_colony_signal(self, global_features: np.ndarray) -> np.ndarray:
        """Генерирует 12-мерный вектор подсказок от Коллективного разума."""
        global_feats_tensor = torch.FloatTensor(global_features).unsqueeze(0).to(self.device)
        signal_tensor, log_prob = self.colony_brain(global_feats_tensor)
        self.last_colony_log_prob = log_prob
        return signal_tensor.squeeze(0).detach().cpu().numpy()

    def select_actions(self, typ: AntType, obs_batch: np.ndarray, masks_batch: np.ndarray) -> List[int]:
        """Выбирает действия для группы муравьёв определённого типа с учётом маски ходов."""
        num_ants = len(obs_batch)
        if num_ants == 0:
            return []

        if random.random() < self.epsilon:
            actions = []
            for i in range(num_ants):
                valid_acts = np.where(masks_batch[i])[0]
                actions.append(int(random.choice(valid_acts)) if len(valid_acts) > 0 else 0)
            return actions

        st = torch.FloatTensor(obs_batch).to(self.device)
        with torch.no_grad():
            q_vals = self.models[typ](st)
            mask_tensor = torch.BoolTensor(masks_batch).to(self.device)
            q_vals[~mask_tensor] = -1e9
            return torch.argmax(q_vals, dim=-1).cpu().numpy().tolist()

    def train_role_step(self, typ: AntType):
        """Обучает Q-сеть конкретной роли на батче из её буфера памяти."""
        memory = self.memories[typ]
        if len(memory) < self.batch_size:
            return

        batch = random.sample(memory, self.batch_size)
        s, a, r, ns, d, m, nm = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(np.array(r)).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(np.array(d)).to(self.device)
        next_masks = torch.BoolTensor(np.array(nm)).to(self.device)

        curr_q = self.models[typ](s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.targets[typ](ns)
            next_q_vals[~next_masks] = -1e9
            max_next_q = next_q_vals.max(dim=1)[0]
            target_q = r + (1 - d) * self.gamma * max_next_q

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizers[typ].zero_grad()
        loss.backward()
        self.optimizers[typ].step()

    def train_colony_brain_step(self, total_team_reward: float):
        if self.last_colony_log_prob is None:
            return

        scaled_reward = total_team_reward * REWARD_SCALE
        self.colony_reward_history.append(scaled_reward)

        # Вычисляем Baseline как среднее значение прошлых эпизодов
        baseline = np.mean(self.colony_reward_history)
        advantage = scaled_reward - baseline

        colony_loss = -self.last_colony_log_prob * advantage

        self.colony_optimizer.zero_grad()
        colony_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.colony_brain.parameters(), max_norm=1.0)
        self.colony_optimizer.step()

    def update_targets(self):
        """Синхронизирует веса целевых сетей."""
        for typ in AntType:
            self.targets[typ].load_state_dict(self.models[typ].state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def save_agent_models(agent: MultiRoleDQNAgent, path_dir: str = "archive", tag: str = ""):
    """Вспомогательная функция для сохранения весов моделей с таймштампом."""
    os.makedirs(path_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{tag}_{timestamp}" if tag else f"{timestamp}"

    torch.save(agent.models[AntType.SCOUT].state_dict(), os.path.join(path_dir, f"ant_brain_scout_{prefix}.pt"))
    torch.save(agent.models[AntType.WORKER].state_dict(), os.path.join(path_dir, f"ant_brain_worker_{prefix}.pt"))
    torch.save(agent.models[AntType.SOLDIER].state_dict(), os.path.join(path_dir, f"ant_brain_soldier_{prefix}.pt"))
    torch.save(agent.colony_brain.state_dict(), os.path.join(path_dir, f"colony_brain_{prefix}.pt"))

    print(f"\n[СХРАНЕНИЕ] Модели успешно сохранены в папку '{path_dir}/' с меткой: {prefix}")


# -------------------------------------------------------------
# 4. Основной цикл обучения (Main Training Loop)
# -------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск обучения на устройстве: {device}")

    env = CompetitiveAntWarsEnv(
        dqn_model_path=r"C:\Dima\Projects\AntWars\src\agent\ant_brain_1785257901.882179.pt",
        single_dqn_model_path=r"C:\Dima\Projects\AntWars\src\agent\one_ant_agent\ant_brain_single_1785268169.9400878.pt"
    )
    agent = MultiRoleDQNAgent(device)

    episodes = 256
    global_step = 0
    start_time = time.time()

    print("=== СТАРТ МУЛЬТИ-РОЛЕВОГО ОБУЧЕНИЯ С КОЛЛЕКТИВНЫМ РАЗУМОМ ===")

    try:
        for ep in range(episodes):
            obs_by_type, info = env.reset()
            masks_by_type = info["masks_by_type"]
            ants_by_type = info["ants_by_type"]

            total_reward = 0.0
            step_cnt = 0

            while True:
                # 1. Глобальный мозг генерирует 12-мерный сигнал для всей колонии
                global_feats = extract_global_features(env)
                colony_signal = agent.get_colony_signal(global_feats)

                # 2. Каждая ролевая модель выбирает действия для своих муравьёв
                actions_by_id = {}
                for typ in AntType:
                    ants = ants_by_type[typ]
                    obs = obs_by_type[typ]
                    masks = masks_by_type[typ]

                    if len(ants) > 0:
                        chosen_acts = agent.select_actions(typ, obs, masks)
                        for i, ant in enumerate(ants):
                            actions_by_id[ant.id] = chosen_acts[i]

                # 3. Делаем шаг в среде с передачей сигнала от Коллективного разума
                next_obs_by_type, rewards_by_id, done, _, next_info = env.step(
                    actions_by_id,
                    colony_signal=colony_signal
                )

                next_masks_by_type = next_info["masks_by_type"]
                next_ants_by_type = next_info["ants_by_type"]

                # 4. Сохраняем опыт в соответствующие ролевые буферы памяти (с масштабированием наград)
                for typ in AntType:
                    ants = ants_by_type[typ]
                    obs = obs_by_type[typ]
                    masks = masks_by_type[typ]

                    next_ants_map = {a.id: i for i, a in enumerate(next_ants_by_type[typ])}

                    for i, ant in enumerate(ants):
                        act = actions_by_id.get(ant.id, 0)
                        r = rewards_by_id.get(ant.id, 0.0)
                        scaled_r = r * REWARD_SCALE  # Нормализуем значение награды

                        if ant.id in next_ants_map:
                            next_idx = next_ants_map[ant.id]
                            next_obs_i = next_obs_by_type[typ][next_idx]
                            next_mask_i = next_masks_by_type[typ][next_idx]
                            done_i = float(done)
                        else:
                            next_obs_i = np.zeros(OBS_DIM[typ], dtype=np.float32)
                            next_mask_i = np.ones(ACTION_DIMS[typ], dtype=bool)
                            done_i = 1.0

                        agent.memories[typ].append(
                            (obs[i], act, scaled_r, next_obs_i, done_i, masks[i], next_mask_i)
                        )

                # 5. Обучаем каждую из ролевых сетей
                if step_cnt % 4 == 0:
                    for typ in AntType:
                        agent.train_role_step(typ)

                obs_by_type = next_obs_by_type
                masks_by_type = next_masks_by_type
                ants_by_type = next_ants_by_type

                total_reward += sum(rewards_by_id.values())
                step_cnt += 1
                global_step += 1

                if global_step % 100 == 0:
                    agent.update_targets()

                if done:
                    agent.train_colony_brain_step(total_reward)
                    break

            agent.decay_epsilon()

            scout_cnt = len(ants_by_type[AntType.SCOUT])
            worker_cnt = len(ants_by_type[AntType.WORKER])
            soldier_cnt = len(ants_by_type[AntType.SOLDIER])

            print(
                f"Эпизод {ep + 1:03d}/{episodes} | "
                f"Награда: {total_reward:7.1f} | "
                f"Счёт: {env.state.score:4d} | "
                f"Юниты (С/Р/Б): {scout_cnt}/{worker_cnt}/{soldier_cnt} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

        print(f"\nОбучение успешно завершено за {time.time() - start_time:.2f} сек!")
        save_agent_models(agent, path_dir="archive", tag="completed")

    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Обучение остановлено пользователем (Ctrl+C)!")
        save_agent_models(agent, path_dir="archive", tag="interrupted")
