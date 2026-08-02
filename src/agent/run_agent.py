import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Добавляем обе папки в пути поиска модулей Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import argparse
import time
from uuid import uuid4
import numpy as np
import torch
from flask.cli import load_dotenv

from src.client.client import AntWarsClient
from fast_env import HEX_DIRECTIONS, MAX_ANTS, MAX_ENEMIES, MAX_FOOD
from src.client.gamestate import GameState
from src.client.visual_map import AsyncVisualizer
from train import QNetwork


MODEL = "ant_brain.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск ИИ-агента AntWars")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=MODEL,
        help="Путь к весам модели (.pt)"
    )
    return parser.parse_args()


def extract_online_obs(state: GameState) -> np.ndarray:
    sq = state.spot.get("q", 0) if state.spot else 0
    sr = state.spot.get("r", 0) if state.spot else 0
    obs = []

    for i in range(MAX_ANTS):
        if i < len(state.ants):
            a = state.ants[i]
            obs.extend([a.q - sq, a.r - sr, a.health, a.type.value, 1.0 if a.food else 0.0])
        else:
            obs.extend([0, 0, 0, -1, 0])

    for i in range(MAX_FOOD):
        if i < len(state.food):
            f = state.food[i]
            obs.extend([f.q - sq, f.r - sr, f.amount])
        else:
            obs.extend([0, 0, 0])

    for i in range(MAX_ENEMIES):
        if i < len(state.enemies):
            e = state.enemies[i]
            obs.extend([e.q - sq, e.r - sr, e.health])
        else:
            obs.extend([0, 0, 0])

    return np.array(obs, dtype=np.float32)


# Добавь функцию генерации маски для реального состояния игры
def get_online_action_mask(state: GameState) -> np.ndarray:
    masks = np.zeros((MAX_ANTS, 7), dtype=bool)

    # Собираем занятые клетки
    occupied = {(a.q, a.r): a for a in state.ants}

    for idx in range(MAX_ANTS):
        if idx >= len(state.ants):
            masks[idx, 0] = True  # Для отсутствующих муравьёв разрешаем только покой
            continue

        ant = state.ants[idx]
        for act_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
            if dq == 0 and dr == 0:
                masks[idx, act_idx] = True
                continue

            tq, tr = ant.q + dq, ant.r + dr
            hex_obj = state.get_hex(tq, tr)

            # 1. Проходимся по стенам/камням
            if hex_obj and (hex_obj.type.value == 5 or hex_obj.cost >= 999):
                continue

            # 2. Не наступаем на сородича того же типа
            blocker = occupied.get((tq, tr))
            if blocker and blocker.type == ant.type:
                continue

            masks[idx, act_idx] = True

    return masks


if __name__ == "__main__":
    load_dotenv()
    BASE_URL = os.getenv("SERVER_URL")
    API_KEY = str(uuid4())

    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        model_path = SCRIPT_DIR / args.model
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Файл модели '{args.model}' не найдён!\n"
            f"Проверено в:\n"
            f"1. {Path(args.model).resolve()}\n"
            f"2. {SCRIPT_DIR / args.model}\n"
            f"3. {PROJECT_ROOT / args.model}"
        )

    if args.model == MODEL:
        MAX_ANTS = 128
        MAX_FOOD = 128
        MAX_ENEMIES = 128

    client = AntWarsClient(BASE_URL, API_KEY)
    input_dim = MAX_ANTS * 5 + MAX_FOOD * 3 + MAX_ENEMIES * 3
    visualizer = AsyncVisualizer('Agent' + ' ' + args.model)

    model = QNetwork(input_dim, num_ants=MAX_ANTS)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Модель нейросети загружена. Подключение к игре...")
    start_wait = client.register()
    if start_wait > 0:
        time.sleep(start_wait + 1)

    while True:
        try:
            state = client.get_arena_state()
            visualizer.update(state)

            obs = extract_online_obs(state)
            action_mask = get_online_action_mask(state)

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                q_vals = model(obs_tensor).squeeze(0)  # Shape: (MAX_ANTS, 7)

                # ЗАМЕНА: зануляем запрещённые ходы
                mask_tensor = torch.BoolTensor(action_mask)
                q_vals[~mask_tensor] = -1e9
                actions = torch.argmax(q_vals, dim=-1).numpy()

            moves = []
            for idx, act in enumerate(actions):
                if idx < len(state.ants):
                    ant = state.ants[idx]
                    dq, dr = HEX_DIRECTIONS[act]
                    if dq != 0 or dr != 0:
                        moves.append({
                            "ant": ant.id,
                            "path": [{"q": ant.q + dq, "r": ant.r + dr}]
                        })

            if moves:
                client.send_moves(moves)

            time.sleep(max(0.1, state.next_turn_in))
        except Exception as ex:
            print("Сессия завершена:", ex)
            break

    visualizer.close()
