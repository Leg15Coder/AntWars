import requests
import time
from typing import *
from gamestate import *

from gamestate import GameState


class DatsPulseClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'X-Auth-Token': f'{self.api_key}'})
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.35:
            time.sleep(0.35 - elapsed)
        self.last_request_time = time.time()

    def get_arena_state(self) -> GameState:
        """Получить текущее состояние игры"""
        self._rate_limit()
        response = self.session.get(f'{self.base_url}/api/arena')
        response.raise_for_status()
        return GameState(response.json())

    def send_moves(self, moves: List[Dict]) -> bool:
        """
        Отправляем приказы муравьям
        moves: Список словарей с 'ant' (ID) и 'path' (список вида {'q': x, 'r': y})
        """
        self._rate_limit()
        payload = {'moves': moves}
        response = self.session.post(f'{self.base_url}/api/move', json=payload)
        if response.status_code == 200:
            return True
        else:
            print(f"Ошибка отправки ходов: {response.status_code} - {response.text}")
            return False

    def get_logs(self) -> Dict:
        self._rate_limit()
        response = self.session.get(f'{self.base_url}/api/logs')
        response.raise_for_status()
        return response.json()

    def register(self) -> int:
        response = self.session.post(f'{self.base_url}/api/register', json=dict())
        if response.status_code == 200:
            print("Регистрация прошла успешно")
            return response.json()["lobbyEndsIn"]
        else:
            print(f"Ошибка регистрации: {response.status_code} - {response.text}")
            return -1

