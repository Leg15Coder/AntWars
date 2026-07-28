import threading
import json
from typing import Set, override
from flask import Flask, request, jsonify
from server_engine import *

app = Flask(__name__)


class MatchServer(FastServerEngine):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

        # Состояние игры
        self.phase = GamePhase.LOBBY
        self.phase_start_time = time.time()
        self.turn_no = 0
        self.last_turn_time = time.time()

        # Данные игроков
        self.registered_tokens: Set[str] = set()

    def get_phase_remaining_time(self) -> float:
        """Возвращает время в секундах до завершения текущей фазы."""
        elapsed = time.time() - self.phase_start_time
        if self.phase == GamePhase.LOBBY:
            return max(0.0, LOBBY_DURATION - elapsed)
        elif self.phase == GamePhase.RUNNING:
            return max(0.0, ROUND_DURATION - elapsed)
        else:
            return max(0.0, BREAK_DURATION - elapsed)

    def register_player(self, token: str) -> Tuple[bool, int, str]:
        """Регистрация игрока в лобби."""
        with self.lock:
            rem_lobby = int(self.get_phase_remaining_time())

            if self.phase == GamePhase.LOBBY and len(self.registered_tokens) < MAX_PLAYERS:
                self.registered_tokens.add(token)
                if token not in self.logs:
                    self.logs[token] = []
                self.logs[token].append(f"Зарегистрирован в лобби. Ожидание старта: {rem_lobby} сек.")
                return True, rem_lobby, "Успешная регистрация в лобби."
            elif self.phase == GamePhase.RUNNING:
                rem_next = rem_lobby + int(BREAK_DURATION) + int(LOBBY_DURATION)
                return False, rem_next, "Игра уже идёт. Ожидайте следующего раунда."
            else:
                rem_next = rem_lobby + int(LOBBY_DURATION)
                return False, rem_next, "Идёт технический перерыв."

    def start_round(self):
        """Запуск раунда совместной игры."""
        if self.history:
            log_filename = os.path.join(log_dir, f"turns_history_at_{int(time.time())}.txt")
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
            self.history.clear()
            self.logs.clear()

        num_teams = len(self.registered_tokens)
        if num_teams < 2:
            self.start_lobby(True)
            return

        self.phase = GamePhase.RUNNING
        self.phase_start_time = time.time()
        self.turn_no = 1
        self.last_turn_time = time.time()

        self.ants.clear()
        self.food_spots.clear()
        self.pending_moves.clear()
        self.teams.clear()

        print(f"\n=== СТАРТ РАУНДА! Игроков: {num_teams} ===")

        map_radius = max(10, 6 + num_teams * 3)
        self.generate_hex_map(map_radius)
        tokens = list(self.registered_tokens)
        dist = max(4, map_radius - 3)

        for idx, token in enumerate(tokens):
            angle = (2 * math.pi * idx) / num_teams
            cq = int(round(dist * math.cos(angle)))
            cr = int(round(dist * math.sin(angle)))

            spot = {"q": cq, "r": cr}
            neighbors = get_hex_neighbors(cq, cr)
            home_hexes = [spot, {"q": neighbors[0][0], "r": neighbors[0][1]}, {"q": neighbors[1][0], "r": neighbors[1][1]}]

            for h in home_hexes:
                self.hex_map[(h["q"], h["r"])] = {"q": h["q"], "r": h["r"], "type": 1, "cost": 1}

            self.teams[token] = {
                "token": token,
                "spot": spot,
                "home": home_hexes,
                "score": 0,
                "nectar": {f"{h['q']},{h['r']}": 0 for h in home_hexes}
            }
            self.logs[token].append(f"Раунд начался! Ваш муравейник на ({cq}, {cr}).")
            self.spawn_ant(token, ant_type=0)

        for _ in range(num_teams * 4):
            self.spawn_random_food()

    def generate_hex_map(self, radius: int):
        self.hex_map.clear()
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                hex_type = 2
                cost = 1
                val = random.random()
                if val < 0.08:
                    hex_type = 3  # грязь
                    cost = 2
                elif val < 0.12:
                    hex_type = 4  # кислота
                    cost = 1
                elif val < 0.16 and (abs(q) > 2 or abs(r) > 2):
                    hex_type = 5  # камень
                    cost = 999

                self.hex_map[(q, r)] = {"q": q, "r": r, "type": hex_type, "cost": cost}

    @override
    def process_turn(self):
        self.last_turn_time = time.time()
        super().process_turn()

    def end_round(self):
        """Завершение раунда."""
        self.phase = GamePhase.BREAK
        self.phase_start_time = time.time()
        print("\n=== РАУНД ЗАВЕРШЁН! ПЕРЕРЫВ ===")

    def start_lobby(self, is_continued: bool = False):
        """Переход к новому лобби."""
        self.phase = GamePhase.LOBBY
        self.phase_start_time = time.time()

        if not is_continued:
            self.registered_tokens.clear()
            print("\n=== ЛОББИ ОТКРЫТО. ОЖИДАНИЕ ИГРОКОВ ===")

    def get_arena_state(self, token: str) -> dict:
        with self.lock:
            if self.phase != GamePhase.RUNNING or token not in self.teams:
                return {
                    "ants": [], "enemies": [], "food": [], "home": [], "map": [],
                    "nextTurnIn": round(self.get_phase_remaining_time(), 1),
                    "score": 0, "spot": {}, "turnNo": self.turn_no
                }

            team = self.teams[token]
            visible_coords = set()

            # Обзор от муравейника (радиус 2)
            visible_coords.update(get_hexes_in_radius(team["spot"]["q"], team["spot"]["r"], 2))

            # Обзор от юнитов
            for ant in self.ants.values():
                if ant["team"] == token and ant["health"] > 0:
                    v_range = UNIT_PROPERTIES[ant["type"]]["view_range"]
                    for vq, vr in ant.get("visited", [(ant["q"], ant["r"])]):
                        visible_coords.update(get_hexes_in_radius(vq, vr, v_range))

            my_ants = [
                {
                    "id": a["id"], "type": a["type"], "q": a["q"], "r": a["r"],
                    "health": a["health"], "food": a["food"], "lastMove": a["lastMove"],
                    "move": a.get("move", []), "lastAttack": a["lastAttack"], "lastEnemyAnt": a["lastEnemyAnt"]
                }
                for a in self.ants.values() if a["team"] == token and a["health"] > 0
            ]

            enemies = [
                {
                    "type": a["type"], "q": a["q"], "r": a["r"], "health": a["health"],
                    "food": a["food"], "attack": UNIT_PROPERTIES[a["type"]]["attack"]
                }
                for a in self.ants.values()
                if a["team"] != token and a["health"] > 0 and (a["q"], a["r"]) in visible_coords
            ]

            return {
                "ants": my_ants,
                "enemies": enemies,
                "food": [f for pos, f in self.food_spots.items() if pos in visible_coords],
                "home": team["home"],
                "map": [h for pos, h in self.hex_map.items() if pos in visible_coords],
                "nextTurnIn": max(0.0, round(TURN_DURATION - (time.time() - self.last_turn_time), 1)),
                "score": team["score"],
                "spot": team["spot"],
                "turnNo": self.turn_no
            }


match_server = MatchServer()


def game_loop_thread():
    """Фоновый цикл автомата состояний."""
    while True:
        time.sleep(0.2)
        with match_server.lock:
            now = time.time()
            elapsed = now - match_server.phase_start_time

            if match_server.phase == GamePhase.LOBBY:
                if elapsed >= LOBBY_DURATION:
                    match_server.start_round()

            elif match_server.phase == GamePhase.RUNNING:
                if elapsed >= ROUND_DURATION:
                    match_server.end_round()
                elif now - match_server.last_turn_time >= TURN_DURATION:
                    match_server.process_turn()

            elif match_server.phase == GamePhase.BREAK:
                if elapsed >= BREAK_DURATION:
                    match_server.start_lobby()


threading.Thread(target=game_loop_thread, daemon=True).start()


# --- HTTP API ---

@app.route('/api/register', methods=['POST'])
def register():
    token = request.headers.get('X-Auth-Token')
    if not token:
        return jsonify({"error": "Missing X-Auth-Token"}), 400

    ok, rem_time, msg = match_server.register_player(token)
    if ok:
        return jsonify({"status": "ok", "lobbyEndsIn": rem_time}), 200
    else:
        return jsonify({"status": "error", "message": msg, "lobbyEndsIn": rem_time}), 400


@app.route('/api/arena', methods=['GET'])
def get_arena():
    token = request.headers.get('X-Auth-Token')
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    state = match_server.get_arena_state(token)
    return jsonify(state), 200


@app.route('/api/move', methods=['POST'])
def make_move():
    token = request.headers.get('X-Auth-Token')
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    moves = (request.get_json() or {}).get("moves", [])
    with match_server.lock:
        match_server.pending_moves[token] = moves

    return jsonify({"status": "ok"}), 200


@app.route('/api/logs', methods=['GET'])
def get_logs():
    token = request.headers.get('X-Auth-Token')
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    with match_server.lock:
        logs = match_server.logs.get(token, [])

    return jsonify({"logs": logs}), 200


if __name__ == "__main__":
    print("Запуск игрового сервера на http://127.0.0.1:8888...")
    app.run(host="127.0.0.1", port=8888, debug=False, use_reloader=False)
