import matplotlib.pyplot as plt
import numpy as np
from models import *
from matplotlib.patches import RegularPolygon
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import zoom
import matplotlib.image as mpimg
from threading import Thread
from queue import Queue
import time
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.raise_window': False})


ant_colors = {
    AntType.WORKER: '#4169e1',  # royal blue
    AntType.SOLDIER: '#32cd32',  # lime green
    AntType.SCOUT: '#00bfff'  # deep sky blue
}

enemy_colors = {
    AntType.WORKER: '#e16941',  # soft orange
    AntType.SOLDIER: '#d2ad32',  # dirty yellow
    AntType.SCOUT: '#ff4f33'  # royal orange
}

hex_colors = {
    HexType.ANTHILL: '#bf00bf',  # purple
    HexType.EMPTY: '#f0f0f0',  # white
    HexType.DIRT: '#00ee60',  # green
    HexType.ACID: '#e2000f',  # red
    HexType.STONE: '#777777'  # gray
}

ANT_SIZE_COEF = 12
MARGIN = 0.5


class AsyncVisualizer:
    def __init__(self):
        self.queue = Queue()
        self.hex_size = 0.5
        self.thread = Thread(target=self._visualization_thread, daemon=True)
        self.food_images = self._load_food_images()
        self.is_running = True

        self.thread.start()

    def _visualization_thread(self):
        plt.ion()  # Включаем интерактивный режим
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.title("Map")
        self.set_figure_position()
        self.ax.set_aspect('equal')

        while self.is_running:
            if not self.queue.empty():
                game_state = self.queue.get()
                self._draw_map(game_state)
                plt.pause(0.1)  # Короткая пауза для обновления графика
            time.sleep(0.2)

    def set_figure_position(self, monitor_num=1):
        """Устанавливает позицию окна на указанном мониторе"""
        try:
            # Получаем размеры экрана
            manager = self.fig.canvas.manager
            if hasattr(manager, 'window'):
                window = manager.window

                # Для Qt backend
                if hasattr(window, 'screen'):
                    screens = window.screen().virtualSiblings()
                    if len(screens) > monitor_num:
                        screen = screens[monitor_num]
                        screen_geometry = screen.geometry()
                        window.setGeometry(screen_geometry)

                # Для Tk backend
                elif hasattr(window, 'winfo_screenwidth'):
                    screen_width = window.winfo_screenwidth()
                    window.geometry(f"+{screen_width + 100}+100")
        except Exception as e:
            print(f"Ошибка позиционирования окна: {e}")

    def _load_food_images(self):
        """Загрузка и предварительная обработка изображений"""
        images = {}
        try:
            # Загрузка с автоматическим масштабированием
            for food_type, path in [
                (FoodType.APPLE, 'img/apple.png'),
                (FoodType.BREAD, 'img/bread.png'),
                (FoodType.NECTAR, 'img/honey.png')
            ]:
                img = mpimg.imread(path)

                # Нормализация размера (макс. 32x32 пикселя)
                scale_factor = min(32 / img.shape[0], 32 / img.shape[1])
                new_height = int(img.shape[0] * scale_factor)
                new_width = int(img.shape[1] * scale_factor)

                # Масштабирование с интерполяцией
                if img.shape[0] > 32 or img.shape[1] > 32:
                    img = zoom(img, (scale_factor, scale_factor, 1))

                images[food_type] = img
        except Exception as e:
            print(f"Ошибка загрузки изображений: {e}")
        return images

    def _draw_map(self, game_state):
        """Функция отрисовки карты (аналогичная предыдущей реализации)"""
        self.ax.clear()
        self.ax.set_title(f"Ход {game_state.turn_no}, очков: {game_state.score}")
        self.ax.set_facecolor('lightgray')

        # Собираем координаты для определения границ
        all_q = [hex.q for hex in game_state.map]
        all_r = [hex.r for hex in game_state.map]

        # Рисуем гексы
        for hex in game_state.map:
            q, r = hex.q, hex.r
            x = q * 0.866  # cos(30°)
            y = r + q * 0.5

            hex_patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=0.5,
                facecolor=hex_colors.get(hex.type, '#ffffff'),
                edgecolor='#888888',  # серый
                linewidth=0.5
            )
            self.ax.add_patch(hex_patch)
            if q % 2 == 0 and r % 2 == 0:
                self.ax.text(x, y - 0.25, f"{q},{r}", ha='center', va='center', fontsize=3, color='black')

        for ant in game_state.ants:
            q, r = ant.q, ant.r
            x = q * 0.866
            y = r + q * 0.5

            if ant.type == AntType.SCOUT:
                x -= 0.2
                y -= 0.2
            elif ant.type == AntType.WORKER:
                y += 0.21
            else:
                x += 0.2
                y -= 0.2

            color = ant_colors.get(ant.type, '#000000')
            self.ax.plot(x, y, marker='o', markersize=self.hex_size*ANT_SIZE_COEF,
                         color=color, markeredgecolor='black', markeredgewidth=0.5)

        for ant in game_state.enemies:
            q, r = ant.q, ant.r
            x = q * 0.866
            y = r + q * 0.5

            if ant.type == AntType.SCOUT:
                x -= 0.2
                y -= 0.2
            elif ant.type == AntType.WORKER:
                y += 0.21
            else:
                x += 0.2
                y -= 0.2

            color = enemy_colors.get(ant.type, '#000000')
            self.ax.plot(x, y, marker='o', markersize=self.hex_size*ANT_SIZE_COEF,
                         color=color, markeredgecolor='black', markeredgewidth=0.5)

        for food in game_state.food:
            q, r = food.q, food.r
            x = q * 0.866
            y = r + q * 0.5

            if food.type in self.food_images:
                imagebox = OffsetImage(self.food_images[food.type], zoom=self.hex_size * 0.9)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                self.ax.add_artist(ab)
                self.ax.text(x + 0.14, y + 0.21, f"{food.amount}",
                        ha='center', va='center', fontsize=7, color='black')

        # Устанавливаем границы с запасом
        if all_q:
            min_x = min(q * 0.866 for q in all_q) - MARGIN
            max_x = max(q * 0.866 for q in all_q) + MARGIN
            min_y = min(r + q * 0.5 for q, r in zip(all_q, all_r)) - MARGIN
            max_y = max(r + q * 0.5 for q, r in zip(all_q, all_r)) + MARGIN

            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y, max_y)
        self.ax.axis('off')

        plt.draw()

    def update(self, game_state):
        """Добавить новое состояние карты в очередь для отрисовки"""
        self.queue.put(game_state)

    def close(self):
        """Завершить работу визуализатора"""
        self.is_running = False
        self.thread.join()
        plt.close(self.fig)


if __name__ == '__main__':
    from client import DatsPulseClient
    from gamestate import *
    API_KEY = "b47dafaf-49a2-4db9-9c68-db6ff36f9cdd"
    # BASE_URL = "http://127.0.0.1:8000"
    BASE_URL = "https://games-test.datsteam.dev"

    client = DatsPulseClient(BASE_URL, API_KEY)
    visualizer = AsyncVisualizer()
    memory = GameMemory()

    start = client.register()
    print(f"Игра начнётся через {start + 1} секунд..." if start > 0 else "Игра уже идёт")
    time.sleep(max(start + 1, 0))

    try:
        while True:
            state = client.get_arena_state()
            print(f"Ход {state.turn_no}, {len(state.ants)} муравьёв, очков: {state.score}")

            visualizer.update(state)

            time_to_wait = max(0, state.next_turn_in)
            if time_to_wait > 0:
                print(f"Ожидайте {time_to_wait:.1f} сек перед следующим ходом...")
                time.sleep(time_to_wait)

            with open(f'logs/log-{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt', 'a', encoding='utf-8') as file:
                file.write(str(client.get_logs()) + '\n')
    except Exception as ex:
        print(ex)
        if "no active game" in str(ex):
            print("Нет активной игры")
        else:
            print("Потеряно соединение")
    finally:
        visualizer.close()
