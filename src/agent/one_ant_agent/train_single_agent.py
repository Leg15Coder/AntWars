import os
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.fast_env import FastAntWarsEnv, HEX_DIRECTIONS, MAX_ANTS

# Константы локального поля зрения муравья
MAX_LOCAL_FRIENDS = 10  # 10 * 5 = 50 фичей
MAX_LOCAL_FOOD = 15     # 15 * 3 = 45 фичей
MAX_LOCAL_ENEMIES = 10  # 10 * 3 = 30 фичей
# Собственные фичи: 5 (rel_q, rel_r, hp, type, has_food)
# Итоговый размер вектора: 5 + 50 + 45 + 30 = 130
SINGLE_OBS_DIM = 130
NUM_ACTIONS = 7


def get_val(obj, key, default=None):
    """Универсальное извлечение значения (работает и со словарями, и с объектами)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    if hasattr(val, "value"):
        return val.value
    return val


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2


def extract_single_ant_obs(state, ant) -> np.ndarray:
    """Формирует локальный вектор наблюдений (130 элементов) для конкретного муравья."""
    spot = get_val(state, "spot") or {"q": 0, "r": 0}
    sq = get_val(spot, "q", 0)
    sr = get_val(spot, "r", 0)

    ant_id = get_val(ant, "id")
    ant_q = get_val(ant, "q", 0)
    ant_r = get_val(ant, "r", 0)
    ant_hp = get_val(ant, "health", 0)
    ant_type = get_val(ant, "type", 0)
    ant_food = get_val(ant, "food")
    has_food = 1.0 if ant_food is not None else 0.0

    obs = []

    # 1. Собственные характеристики муравья (5)
    obs.extend([ant_q - sq, ant_r - sr, ant_hp, ant_type, has_food])

    # 2. Ближайшие союзники (10 x 5 = 50)
    all_ants = get_val(state, "ants") or []
    friends = []
    for friend in all_ants:
        f_id = get_val(friend, "id")
        if f_id != ant_id:
            fq = get_val(friend, "q", 0)
            fr = get_val(friend, "r", 0)
            dist = hex_distance(ant_q, ant_r, fq, fr)
            friends.append((dist, friend))
    friends.sort(key=lambda x: x[0])

    for i in range(MAX_LOCAL_FRIENDS):
        if i < len(friends):
            _, f = friends[i]
            fq = get_val(f, "q", 0)
            fr = get_val(f, "r", 0)
            f_hp = get_val(f, "health", 0)
            f_tp = get_val(f, "type", 0)
            f_fd = 1.0 if get_val(f, "food") is not None else 0.0
            obs.extend([fq - ant_q, fr - ant_r, f_hp, f_tp, f_fd])
        else:
            obs.extend([0, 0, 0, -1, 0])

    # 3. Ближайшая еда (15 x 3 = 45)
    all_food = get_val(state, "food") or []
    foods = []
    for food in all_food:
        fdq = get_val(food, "q", 0)
        fdr = get_val(food, "r", 0)
        dist = hex_distance(ant_q, ant_r, fdq, fdr)
        foods.append((dist, food))
    foods.sort(key=lambda x: x[0])

    for i in range(MAX_LOCAL_FOOD):
        if i < len(foods):
            _, fd = foods[i]
            fdq = get_val(fd, "q", 0)
            fdr = get_val(fd, "r", 0)
            amount = get_val(fd, "amount", 0)
            obs.extend([fdq - ant_q, fdr - ant_r, amount])
        else:
            obs.extend([0, 0, 0])

    # 4. Ближайшие враги (10 x 3 = 30)
    all_enemies = get_val(state, "enemies") or []
    enemies = []
    for enemy in all_enemies:
        eq = get_val(enemy, "q", 0)
        er = get_val(enemy, "r", 0)
        dist = hex_distance(ant_q, ant_r, eq, er)
        enemies.append((dist, enemy))
    enemies.sort(key=lambda x: x[0])

    for i in range(MAX_LOCAL_ENEMIES):
        if i < len(enemies):
            _, e = enemies[i]
            eq = get_val(e, "q", 0)
            er = get_val(e, "r", 0)
            ehp = get_val(e, "health", 0)
            obs.extend([eq - ant_q, er - ant_r, ehp])
        else:
            obs.extend([0, 0, 0])

    return np.array(obs, dtype=np.float32)


class SingleAntQNetwork(nn.Module):
    def __init__(self, input_dim: int = SINGLE_OBS_DIM, num_actions: int = NUM_ACTIONS):
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


class SingleAntDQNAgent:
    def __init__(self, input_dim: int = SINGLE_OBS_DIM):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SingleAntQNetwork(input_dim).to(self.device)
        self.target = SingleAntQNetwork(input_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.993
        self.batch_size = 128

    def act(self, obs_vec: np.ndarray, action_mask: np.ndarray) -> int:
        if random.random() < self.epsilon:
            valid_acts = np.where(action_mask)[0]
            return int(random.choice(valid_acts)) if len(valid_acts) > 0 else 0

        st = torch.FloatTensor(obs_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(st).squeeze(0)
            mask_tensor = torch.BoolTensor(action_mask).to(self.device)
            q_vals[~mask_tensor] = -1e9
            return int(torch.argmax(q_vals).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d, m, nm = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(np.array(r)).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(np.array(d)).to(self.device)
        next_masks = torch.BoolTensor(np.array(nm)).to(self.device)

        curr_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target(ns)
            next_q_vals[~next_masks] = -1e9
            max_next_q = next_q_vals.max(dim=1)[0]
            target_q = r + (1 - d) * self.gamma * max_next_q

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = FastAntWarsEnv(num_bots=3)
    agent = SingleAntDQNAgent(SINGLE_OBS_DIM)

    episodes = 256
    global_step = 0
    start_time = time.time()

    print("=== СТАРТ ОБУЧЕНИЯ (ОДИН МУРАВЕЙ - ОДИН АГЕНТ) ===")

    for ep in range(episodes):
        obs, info = env.reset()
        action_masks = info["action_mask"]
        total_reward = 0
        step_cnt = 0

        while True:
            state_before = env.engine.get_state(env.agent_token)
            my_ants_before = state_before["ants"]
            spot_before = get_val(state_before, "spot") or {"q": 0, "r": 0}
            sq_b, sr_b = get_val(spot_before, "q", 0), get_val(spot_before, "r", 0)

            num_ants_before = min(len(my_ants_before), MAX_ANTS)
            ant_obs_before = []
            actions_chosen = []

            # 1. Принимаем решение для каждого живого муравья по отдельности
            for i in range(num_ants_before):
                ant = my_ants_before[i]
                obs_i = extract_single_ant_obs(state_before, ant)
                mask_i = action_masks[i]
                act_i = agent.act(obs_i, mask_i)

                ant_obs_before.append(obs_i)
                actions_chosen.append(act_i)

            # 2. Формируем единый массив действий для среды
            full_actions = np.zeros(MAX_ANTS, dtype=int)
            full_actions[:num_ants_before] = actions_chosen

            # 3. Делаем шаг в среде
            _, rewards_env, done, _, next_info = env.step(full_actions)
            next_action_masks = next_info["action_mask"]

            state_after = env.engine.get_state(env.agent_token)
            ants_after_map = {a["id"]: (idx, a) for idx, a in enumerate(state_after["ants"])}

            # 4. Сохраняем индивидуальный опыт каждого муравья
            for i in range(num_ants_before):
                ant_b = my_ants_before[i]
                ant_id = ant_b["id"]
                obs_i = ant_obs_before[i]
                act_i = actions_chosen[i]
                r_i = float(rewards_env[i])
                mask_i = action_masks[i]

                # Небольшой штраф за покой (действие 0), если муравей не сдаёт ресурс на муравейнике
                if act_i == 0:
                    is_home = (ant_b["q"] == sq_b and ant_b["r"] == sr_b)
                    if ant_b["food"] is None or not is_home:
                        r_i -= 0.02

                # Если муравей выжил после хода
                if ant_id in ants_after_map:
                    idx_after, ant_a = ants_after_map[ant_id]
                    next_obs_i = extract_single_ant_obs(state_after, ant_a)
                    next_mask_i = next_action_masks[idx_after]
                    done_i = float(done)
                else:
                    # Если муравей погиб
                    next_obs_i = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
                    next_mask_i = np.ones(NUM_ACTIONS, dtype=bool)
                    done_i = 1.0

                agent.memory.append((obs_i, act_i, r_i, next_obs_i, done_i, mask_i, next_mask_i))

            # 5. Обучаем сеть
            if step_cnt % 4 == 0:
                agent.train_step()

            action_masks = next_action_masks
            total_reward += np.sum(rewards_env[:num_ants_before])
            step_cnt += 1
            global_step += 1

            if global_step % 100 == 0:
                agent.target.load_state_dict(agent.model.state_dict())

            if done:
                break

        agent.decay_epsilon()
        print(f"Эпизод {ep + 1:03d}/{episodes} | Награда: {total_reward:6.1f} | Epsilon: {agent.epsilon:.2f} | Память: {len(agent.memory)}")

    print(f"\nОбучение завершено за {time.time() - start_time:.2f} сек!")
    file = f"ant_brain_single_{time.time()}.pt"
    torch.save(agent.model.state_dict(), file)
    print(f"Веса модели успешно сохранены в '{file}'.")
