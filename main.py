from datetime import datetime

from algorithm import *
from client import *
from visual_map import AsyncVisualizer
import threading
import time


def timer(wait_seconds: float) -> None:
    pass


if __name__ == "__main__":
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    # BASE_URL = "http://127.0.0.1:8000"
    BASE_URL = "https://games.datsteam.dev"
    # BASE_URL = "https://games-test.datsteam.dev"

    client = DatsPulseClient(BASE_URL, API_KEY)
    strategy = Strategy(client)
    visualizer = AsyncVisualizer()
    memory = GameMemory()

    start = client.register()
    print(f"Игра начнётся через {start + 1} секунд..." if start > 0 else "Игра уже идёт")
    time.sleep(max(start + 1, 0))

    while True:
        state = client.get_arena_state()
        state.memory = memory
        print(f"Ход {state.turn_no}, {len(state.ants)} муравьёв, очков: {state.score}")

        if not memory.initialized:
            memory.init_home(state)
        memory.update(state)

        visualizer.update(state)

        strategy.make_turn(state)
        time_to_wait = max(0, state.next_turn_in)
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        with open(f'logs/log-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt', 'a', encoding='utf-8') as file:
            file.write(str(client.get_logs()) + '\n')
    print(ex)
    if "no active game" in str(ex):
        print("Нет активной игры")
    else:
        print("Потеряно соединение")
    visualizer.close()
