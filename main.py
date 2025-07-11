from datetime import datetime

from algorithm import *
from client import *
from visual_map import AsyncVisualizer
import threading
import time


if __name__ == "__main__":
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    BASE_URL = "https://games-test.datsteam.dev"

    client = DatsPulseClient(BASE_URL, API_KEY)
    strategy = Strategy(client)
    visualizer = AsyncVisualizer()

    start = client.register()
    print(f"Игра начнётся через {start + 1} секунд..." if start > 0 else "Игра уже идёт")
    time.sleep(max(start + 2, 0))

    #try:
    while True:
        state = client.get_arena_state()
        print(f"Ход {state.turn_no}, {len(state.ants)} муравьёв, очков: {state.score}")

        visualizer.update(state)

        strategy.make_turn()
        time_to_wait = max(0, state.next_turn_in)
        if time_to_wait > 0:
            print(f"Ожидайте {time_to_wait:.1f} сек перед следующим ходом...")
            time.sleep(time_to_wait)

        with open(f'logs/log-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt', 'a', encoding='utf-8') as file:
            file.write(str(client.get_logs()) + '\n')
#except Exception as ex:
    print(ex)
    if "no active game" in str(ex):
        print("Нет активной игры")
    else:
        print("Потеряно соединение")
#finally:
    visualizer.close()
