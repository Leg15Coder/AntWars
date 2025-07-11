import matplotlib.pyplot as plt
import numpy as np
from models import *
from matplotlib.patches import RegularPolygon


class HexMapVisualizer:
    @staticmethod
    def draw_map(game_state, title="DatsPulse Map"):
        """
        визуализация гексагональной карты без предупреждений

        Параметры:
            game_state: объект GameState с данными о карте
            title: заголовок графика
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.set_title(title, pad=20)
        ax.set_facecolor('lightgray')

        # Цвета для разных типов гексов
        hex_colors = {
            HexType.ANTHILL: '#ff9999',  # светло-красный
            HexType.EMPTY: '#f0f0f0',  # очень светлый серый
            HexType.DIRT: '#d2b48c',  # tan цвет
            HexType.ACID: '#90ee90',  # светло-зеленый
            HexType.STONE: '#333333'  # темно-серый
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
            ax.add_patch(hex_patch)

            # Подписываем только основные координаты для уменьшения нагромождения
            if abs(q) <= 2 or abs(r) <= 2 or q == 0 or r == 0:
                ax.text(x, y, f"{q},{r}", ha='center', va='center', fontsize=6, color='#555555')

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
            color = ant_colors.get(ant.type, '#000000')
            ax.plot(x, y, marker='o', markersize=8, color=color, markeredgecolor='black', markeredgewidth=0.5)
            ax.text(x, y + 0.25, f"{ant.type.name[:1]}", ha='center', va='center',
                    fontsize=8, weight='bold', color='white')

        # Рисуем ресурсы
        food_symbols = {
            FoodType.APPLE: 'A',
            FoodType.BREAD: 'B',
            FoodType.NECTAR: 'N'
        }

        for food in game_state.food:
            q, r = food.q, food.r
            x = q * 0.866
            y = r + q * 0.5
            symbol = food_symbols.get(food.type, '?')
            ax.text(x, y - 0.25, f"{symbol}{food.amount}",
                    ha='center', va='center', fontsize=10)

        # Устанавливаем границы с запасом
        margin = 1.5
        min_x = min(q * 0.866 for q in all_q) - margin
        max_x = max(q * 0.866 for q in all_q) + margin
        min_y = min(r + q * 0.5 for q, r in zip(all_q, all_r)) - margin
        max_y = max(r + q * 0.5 for q, r in zip(all_q, all_r)) + margin

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.axis('off')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
