from datetime import datetime

from algorithm import *
from client import *
from visual_map import HexMapVisualizer
import time


if __name__ == "__main__":
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    BASE_URL = "https://games-test.datsteam.dev"

    client = DatsPulseClient(BASE_URL, API_KEY)
    strategy = Strategy(client)

    start = client.register()
    print(f"Игра начнётся через {start + 1} секунд..." if start > 0 else "Игра уже идёт")
    time.sleep(max(start + 2, 0))

    while True:
        state = client.get_arena_state()
        print(f"Ход {state.turn_no}, {len(state.ants)} муравьёв, очков: {state.score}")
        HexMapVisualizer.draw_map(state, "Текущее состояние игры")
        print(1)
        strategy.make_turn()
        print(2)
        time_to_wait = max(0, state.next_turn_in - 0.1)
        print(3)
        if time_to_wait > 0:
            print(f"Ожидайте {time_to_wait:.1f} сек перед следующим ходом...")
            time.sleep(time_to_wait)

        with open(f'logs/log-{datetime.now()}.txt', 'w') as file:
            file.write(client.get_logs())
