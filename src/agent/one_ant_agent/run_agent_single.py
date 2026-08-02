import os
import time
from uuid import uuid4
import numpy as np
import torch
from flask.cli import load_dotenv

from src.client.client import AntWarsClient
from src.client.gamestate import GameState
from src.client.visual_map import AsyncVisualizer
from train_single_agent import (
    SINGLE_OBS_DIM,
    HEX_DIRECTIONS,
    SingleAntQNetwork,
    extract_single_ant_obs,
)

MODEL_FILE = "ant_brain_single_1785268169.9400878.pt"


def get_single_ant_mask_online(state: GameState, ant) -> np.ndarray:
    """Генерирует маску допустимых ходов (размер 7) для конкретного муравья в реальном матче."""
    mask = np.zeros(7, dtype=bool)
    mask[0] = True  # Стоять на месте разрешено всегда

    occupied = {(a.q, a.r): a for a in state.ants}
    for e in state.enemies:
        occupied[(e.q, e.r)] = e

    for act_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
        if dq == 0 and dr == 0:
            continue

        tq, tr = ant.q + dq, ant.r + dr
        hex_obj = state.get_hex(tq, tr)

        # Проверка на камни/стены
        if hex_obj and (hex_obj.type.value == 5 or hex_obj.cost >= 999):
            continue

        # Проверка на занятость клетки сородичем того же типа или врагом
        blocker = occupied.get((tq, tr))
        if blocker:
            is_friendly = hasattr(blocker, "id")
            if not is_friendly or blocker.type == ant.type:
                continue

        mask[act_idx] = True

    return mask


if __name__ == "__main__":
    load_dotenv()
    BASE_URL = os.getenv("SERVER_URL")
    API_KEY = str(uuid4())

    client = AntWarsClient(BASE_URL, API_KEY)
    visualizer = AsyncVisualizer("RL Single-Ant Agent")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleAntQNetwork(SINGLE_OBS_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    print("Модель локального агента загружена. Подключение к серверу...")
    start_wait = client.register()
    if start_wait > 0:
        time.sleep(start_wait + 1)

    while True:
        try:
            state = client.get_arena_state()
            visualizer.update(state)

            moves = []

            # Проходимся по каждому нашему муравью
            for ant in state.ants:
                ant_obs = extract_single_ant_obs(state, ant)
                action_mask = get_single_ant_mask_online(state, ant)

                obs_tensor = torch.FloatTensor(ant_obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_vals = model(obs_tensor).squeeze(0)
                    mask_tensor = torch.BoolTensor(action_mask).to(device)
                    q_vals[~mask_tensor] = -1e9
                    act = int(torch.argmax(q_vals).item())

                # Если выбрано движение
                if act != 0:
                    dq, dr = HEX_DIRECTIONS[act]
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
