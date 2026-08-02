import os
import sys
import time
import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import numpy as np
import torch
from dotenv import load_dotenv

# Настройка путей для импорта модулей
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ARCHIVE_DIR = SCRIPT_DIR / "archive"

from src.client.client import AntWarsClient
from src.client.gamestate import GameState
from src.client.models import AntType, Ant, HexType, EnemyAnt
from src.client.visual_map import AsyncVisualizer
from hex_pathfinding import astar_hex_path

# Импортируем архитектуру из нашего модуля обучения
from src.agent.my_agent.may_agent_env import (
    OBS_DIM,
    ACTION_DIMS,
    COLONY_SIGNAL_DIM,
    MOVE_DIRECTIONS,
    VISION_DIRECTIONS,
    MAX_OP,
    HEX_NEIGHBORS
)
from src.agent.my_agent.my_agent_train import RoleQNetwork, ColonyBrainNetwork

POSTFIX = '_completed_20260729_220459'

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск ИИ-агента AntWars с Коллективным Разумом")
    parser.add_argument("--model-dir", "-d", type=str, default=".", help="Директория с .pt файлами весов")
    parser.add_argument("--no-vis", action="store_true", help="Отключить визуализатор")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=POSTFIX,
        help="Путь к весам модели (.pt)"
    )
    return parser.parse_args()


def extract_global_features(state: GameState, visited_count: int) -> np.ndarray:
    """Формирует 16-мерный вектор глобального состояния для Коллективного Разума."""
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
        float(state.turn_no),
        float(visited_count),
        0.0, 0.0, 0.0, 0.0, 0.0  # Паддинг до 16 элементов
    ]
    return np.array(features[:16], dtype=np.float32)


def extract_ant_obs(ant: Ant, state: GameState, colony_signal: np.ndarray) -> np.ndarray:
    """Извлекает вектор наблюдений для одного муравья в реальной игре."""
    spot = state.spot if state.spot else {"q": 0, "r": 0}
    sq = spot.get("q", 0) if isinstance(spot, dict) else getattr(spot, "q", 0)
    sr = spot.get("r", 0) if isinstance(spot, dict) else getattr(spot, "r", 0)

    typ = ant.type
    ant_q, ant_r = ant.q, ant.r
    has_food_flag = 1.0 if ant.food is not None else 0.0

    obs = [ant_q - sq, ant_r - sr, float(ant.health), has_food_flag]

    for dq, dr in VISION_DIRECTIONS[typ]:
        tq, tr = ant_q + dq, ant_r + dr
        th = state.get_hex(tq, tr)

        if th is None:
            obs.extend([-1.0, 0.0, -1.0, -1.0, 0.0])
            continue

        # Еда
        if th.food is not None:
            f_type = th.food.type.value if hasattr(th.food.type, "value") else th.food.type
            obs.extend([float(f_type), float(th.food.amount)])
        else:
            obs.extend([-1.0, 0.0])

        # Тип гекса
        h_type = th.type.value if hasattr(th.type, "value") else th.type
        obs.append(float(h_type))

        # Сущность на клетке
        if th.entity is not None:
            is_enemy = 1.0 if isinstance(th.entity, EnemyAnt) else 0.0
            hp = float(getattr(th.entity, "health", 0))
            obs.extend([is_enemy, hp])
        else:
            obs.extend([-1.0, 0.0])

    obs.extend(colony_signal.tolist())
    return np.array(obs, dtype=np.float32)


def get_ant_action_mask(ant: Ant, state: GameState) -> np.ndarray:
    """Генерирует маску допустимых ходов для конкретного муравья."""
    typ = ant.type
    dirs = MOVE_DIRECTIONS[typ]
    mask = np.zeros(len(dirs), dtype=bool)

    occupied_my = {(a.q, a.r): a for a in state.ants}
    occupied_enemies = {(e.q, e.r): e for e in getattr(state, "enemies", [])}

    for j, (dq, dr) in enumerate(dirs):
        if dq == 0 and dr == 0:
            mask[j] = True
            continue

        tq, tr = ant.q + dq, ant.r + dr

        if state.is_border(tq, tr):
            continue

        th = state.get_hex(tq, tr)
        # Если клетка известна и это камень или непроходимое препятствие
        if th is not None and (th.type == HexType.STONE or getattr(th, "cost", 1) >= 999):
            continue

        # Нельзя наступать на врага
        if (tq, tr) in occupied_enemies:
            continue

        # Нельзя наступать на союзника того же типа
        blocker = occupied_my.get((tq, tr))
        if blocker and blocker.id != ant.id and blocker.type == ant.type:
            continue

        mask[j] = True

    return mask


def axial_hex_distance(dq: int, dr: int) -> int:
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def main():
    load_dotenv()
    args = parse_args()
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация агента на устройстве: {device}")

    # 1. Загрузка нейросетей
    role_models = {}
    for typ, name in [(AntType.SCOUT, "scout"), (AntType.WORKER, "worker"), (AntType.SOLDIER, "soldier")]:
        path = model_dir / f"ant_brain_{name}{args.model}.pt"
        if not path.exists():
            path = SCRIPT_DIR / f"ant_brain_{name}{args.model}.pt"

        model = RoleQNetwork(OBS_DIM[typ], ACTION_DIMS[typ]).to(device)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Загружены веса для роли {name.capitalize()}: {path}")
        else:
            print(f"ВНИМАНИЕ: Файл {path} не найден! Использована случайная инициализация.")
        model.eval()
        role_models[typ] = model

    colony_brain_path = model_dir / f"colony_brain{args.model}.pt"
    if not colony_brain_path.exists():
        colony_brain_path = SCRIPT_DIR / f"colony_brain{args.model}.pt"

    colony_brain = ColonyBrainNetwork(global_input_dim=16, signal_dim=COLONY_SIGNAL_DIM).to(device)
    if colony_brain_path.exists():
        colony_brain.load_state_dict(torch.load(colony_brain_path, map_location=device))
        print(f"Загружены веса Коллективного Разума: {colony_brain_path}")
    else:
        print(f"ВНИМАНИЕ: Файл {colony_brain_path} не найден! Использована случайная инициализация.")
    colony_brain.eval()

    # 2. Подключение к серверу
    base_url = os.getenv("SERVER_URL", "http://antwars.duckdns.org")
    api_key = str(uuid4())
    client = AntWarsClient(base_url, api_key)

    visualizer = None
    if not args.no_vis:
        visualizer = AsyncVisualizer(f"Multi-Role RL Agent{args.model}")

    print("Регистрация в раунде...")
    start_wait = client.register()
    if start_wait > 0:
        print(f"Ожидание старта игры: {start_wait + 1} сек...")
        time.sleep(start_wait + 1)

    visited_hexes = set()
    logs_dir = os.getenv("LOGS_PATH", "./logs")
    os.makedirs(logs_dir, exist_ok=True)

    print("=== АГЕНТ УСПЕШНО ЗАПУЩЕН И ГОТОВ К ИГРЕ ===")

    try:
        while True:
            # Получение состояния арены
            state = client.get_arena_state()
            if visualizer:
                visualizer.update(state)

            # Фиксируем посещённые клетки
            for ant in state.ants:
                visited_hexes.add((ant.q, ant.r))

            print(
                f"Ход {state.turn_no:03d} | "
                f"Очки: {state.score:4d} | "
                f"Юнитов: {len(state.ants)} | "
                f"До след. хода: {state.next_turn_in:.2f}с"
            )

            # 1. Генерируем вектор сигнала от Коллективного Разума
            global_feats = extract_global_features(state, len(visited_hexes))
            global_tensor = torch.FloatTensor(global_feats).unsqueeze(0).to(device)
            with torch.no_grad():
                colony_signal, _ = colony_brain(global_tensor)
                colony_signal = colony_signal.squeeze(0).cpu().numpy()

            moves = []

            # 2. Принимаем решения для каждого муравья
            for ant in state.ants:
                obs = extract_ant_obs(ant, state, colony_signal)
                mask = get_ant_action_mask(ant, state)

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_tensor = torch.BoolTensor(mask).to(device)

                with torch.no_grad():
                    q_vals = role_models[ant.type](obs_tensor).squeeze(0)
                    q_vals[~mask_tensor] = -1e9
                    act_idx = int(torch.argmax(q_vals).item())

                dirs = MOVE_DIRECTIONS[ant.type]
                dq, dr = dirs[act_idx]

                if dq == 0 and dr == 0:
                    continue

                target_hex = (ant.q + dq, ant.r + dr)
                max_op = MAX_OP[ant.type][0]

                # Поиск пути через A*
                full_path = astar_hex_path(
                    start=(ant.q, ant.r),
                    target=target_hex,
                    max_op=max_op,
                    game_state=state
                )

                # Фолллбэк: шаг в ближайшую доступную смежную клетку
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
                        game_state=state
                    )

                if full_path:
                    moves.append({
                        "ant": ant.id,
                        "path": full_path
                    })

            # Отправляем ходы на сервер
            if moves:
                client.send_moves(moves)

            # Логирование
            log_file = os.path.join(logs_dir, f"log-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt")
            try:
                logs = client.get_logs()
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"Turn {state.turn_no}: {logs}\n")
            except Exception:
                pass

            time_to_wait = max(0.05, state.next_turn_in)
            time.sleep(time_to_wait)

    except Exception as ex:
        print("Сессия игры завершена или перервана:", ex)
        if "no active game" in str(ex):
            print("Активная игра не найдена.")
    finally:
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    main()
