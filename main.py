from algorithm import *
from client import *
import time


if __name__ == "__main__":
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    BASE_URL = "https://games.datsteam.dev/api/datspulse"

    client = DatsPulseClient(BASE_URL, API_KEY)
    strategy = Strategy(client)

    while True:
        state = client.get_arena_state()
        print(f"Ход {state.turn_no}, {len(state.ants)} муравьёв, очков: {state.score}")

        strategy.make_turn()

        time_to_wait = max(0, state.next_turn_in - 0.1)
        if time_to_wait > 0:
            print(f"Ожидайте {time_to_wait:.1f} сек перед следующим ходом...")
            time.sleep(time_to_wait)
