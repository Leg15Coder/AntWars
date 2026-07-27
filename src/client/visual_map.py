import matplotlib.pyplot as plt
import numpy as np
from src.client.models import *
from matplotlib.patches import RegularPolygon
from matplotlib.font_manager import FontProperties
from threading import Thread
from queue import Queue
import time


class AsyncVisualizer:
    def __init__(self, title: str ='GAME'):
        self.queue = Queue()
        self.thread = Thread(target=self._visualization_thread, daemon=True)
        self.thread.start()
        self.emoji_font = self._find_emoji_font()
        self.is_running = True
        self.title = title

    def _visualization_thread(self):
        plt.ion()  # Включаем интерактивный режим
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')

        while self.is_running:
            if not self.queue.empty():
                game_state = self.queue.get()
                self._draw_map(game_state)
                plt.pause(0.1)  # Короткая пауза для обновления графика
            time.sleep(0.1)  # Проверяем очередь 10 раз в секунду

    def _find_emoji_font(self):
        """Пытаемся найти шрифт с поддержкой эмодзи"""
        try:
            # Альтернативный вариант для Windows/Linux
            return FontProperties(fname='/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf')
        except:
            # Если не нашли, используем системный шрифт (эмодзи могут отображаться как квадратики)
            return FontProperties(family='sans-serif')

    def _draw_map(self, game_state):
        """Функция отрисовки карты (аналогичная предыдущей реализации)"""
        self.ax.clear()
        self.ax.set_title(f"{self.title} - Turn {game_state.turn_no}")
        self.ax.set_facecolor('lightgray')

        # Цвета для разных типов гексов
        hex_colors = {
            HexType.ANTHILL: '#68006C',
            HexType.EMPTY: '#f0f0f0',
            HexType.DIRT: '#007730',
            HexType.ACID: '#A2000C',
            HexType.STONE: '#333333'
        }

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

            # Подписываем только основные координаты для уменьшения нагромождения
            if abs(q) <= 2 or abs(r) <= 2 or q == 0 or r == 0:
                self.ax.text(x, y, f"{q},{r}", ha='center', va='center', fontsize=6, color='#555555')

        # Рисуем муравьев
        ant_colors = {
            AntType.WORKER: '#4169e1',  # royal blue
            AntType.SOLDIER: '#32cd32',  # lime green
            AntType.SCOUT: '#00bfff'  # deep sky blue
        }

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
            self.ax.plot(x, y, marker='o', markersize=8, color=color, markeredgecolor='black', markeredgewidth=0.5)

        # Рисуем ресурсы
        food_symbols = {
            FoodType.APPLE: '🍎',
            FoodType.BREAD: '🍞',
            FoodType.NECTAR: '🍯'
        }

        for food in game_state.food:
            q, r = food.q, food.r
            x = q * 0.866
            y = r + q * 0.5
            symbol = food_symbols.get(food.type, '?')
            self.ax.text(x, y - 0.25, f"{symbol}{food.amount}",
                    ha='center', va='center', fontsize=8)

        # Устанавливаем границы с запасом
        margin = 1.5
        min_x = min(q * 0.866 for q in all_q) - margin
        max_x = max(q * 0.866 for q in all_q) + margin
        min_y = min(r + q * 0.5 for q, r in zip(all_q, all_r)) - margin
        max_y = max(r + q * 0.5 for q, r in zip(all_q, all_r)) + margin

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
