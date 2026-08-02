import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.fast_env import FastAntWarsEnv, MAX_ANTS


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_ants: int = MAX_ANTS, num_actions: int = 7):
        super().__init__()
        self.num_ants = num_ants
        self.num_actions = num_actions
        self.base = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        # Ускоренная голова: один вектор выходов вместо 128 отдельных слоёв
        self.head = nn.Linear(1024, num_ants * num_actions)

    def forward(self, x):
        features = self.base(x)
        q_vals = self.head(features)
        return q_vals.view(-1, self.num_ants, self.num_actions)


class DQNAgent:
    def __init__(self, state_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_dim).to(self.device)
        self.target = QNetwork(state_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.06
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def act(self, state: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        if random.random() < self.epsilon:
            actions = np.zeros(MAX_ANTS, dtype=int)
            for i in range(MAX_ANTS):
                valid_acts = np.where(action_mask[i])[0]
                actions[i] = random.choice(valid_acts) if len(valid_acts) > 0 else 0
            return actions

        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(st).squeeze(0)
            mask_tensor = torch.BoolTensor(action_mask).to(self.device)
            q_vals[~mask_tensor] = -1e9
            return torch.argmax(q_vals, dim=-1).cpu().numpy()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d, m, nm = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(np.array(r)).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(-1).to(self.device)
        next_masks = torch.BoolTensor(np.array(nm)).to(self.device)

        curr_q = self.model(s).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_vals = self.target(ns)
            next_q_vals[~next_masks] = -1e9
            max_next_q = next_q_vals.max(dim=-1)[0]
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
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_dim)

    global_step = 0
    episodes = 512
    start_time = time.time()

    print("=== СТАРТ УСКОРЕННОГО ОБУЧЕНИЯ RL-АГЕНТА ===")
    for ep in range(episodes):
        obs, info = env.reset()
        action_mask = info["action_mask"]
        total_reward = 0
        step_cnt = 0

        while True:
            action = agent.act(obs, action_mask)
            next_obs, reward, done, _, next_info = env.step(action)
            next_action_mask = next_info["action_mask"]

            agent.memory.append((obs, action, reward, next_obs, done, action_mask, next_action_mask))

            if step_cnt % 4 == 0:
                agent.train_step()

            obs = next_obs
            action_mask = next_action_mask  # ИСПРАВЛЕНО: обновляем маску для следующего шага

            total_reward += np.sum(reward)
            step_cnt += 1
            global_step += 1

            if global_step % 100 == 0:
                agent.target.load_state_dict(agent.model.state_dict())

            if done:
                break

        # Уменьшаем исследующий коэффициент строго в конце эпизода
        agent.decay_epsilon()

        print(f"Эпизод {ep + 1:03d}/{episodes} | Награда: {total_reward:6.1f} | Epsilon: {agent.epsilon:.2f}")

    print(f"\nОбучение завершено за {time.time() - start_time:.2f} сек!")
    file = f"ant_brain_{time.time()}.pt"
    torch.save(agent.model.state_dict(), file)
    print(f"Веса модели сохранены в '{file}'.")
