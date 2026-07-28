import pygame
import math
import os
import threading
from queue import Queue, Empty
from typing import Optional, Tuple, List, Dict
from src.client.models import HexType, AntType, FoodType
from src.client.gamestate import GameState


class AsyncVisualizer:
    def __init__(self, title: str = 'AntWars Game'):
        self.title = title
        self.queue = Queue()
        self.is_running = True

        # Запускаем Pygame в отдельном потоке
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def update(self, game_state: GameState):
        """Передаём новое состояние игры в очередь для отрисовки."""
        if self.is_running:
            self.queue.put(game_state)

    def close(self):
        """Завершаем работу визуализатора."""
        self.is_running = False

    def _run(self):
        pygame.init()
        pygame.font.init()

        screen_width, screen_height = 1100, 800
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption(self.title)

        clock = pygame.time.Clock()

        # Шрифты
        font_small = pygame.font.SysFont('Consolas', 11)
        font_medium = pygame.font.SysFont('Arial', 14, bold=True)
        font_large = pygame.font.SysFont('Arial', 18, bold=True)

        # Камера (панорамирование и зум)
        camera_x = screen_width // 2
        camera_y = screen_height // 2
        zoom = 28.0  # Радиус гекса в пикселях

        dragging = False
        drag_start = (0, 0)

        game_state: Optional[GameState] = None

        # Цветовая палитра для типов гексов
        hex_colors = {
            HexType.ANTHILL: (106, 27, 154),   # Пурпурный
            HexType.EMPTY: (220, 220, 220),    # Светло-серый
            HexType.DIRT: (46, 125, 50),       # Зелёный
            HexType.ACID: (198, 40, 40),       # Красный
            HexType.STONE: (66, 66, 66)        # Тёмно-серый
        }

        # Цвета муравьёв
        ant_colors = {
            AntType.WORKER: (33, 150, 243),    # Синий
            AntType.SOLDIER: (76, 175, 80),    # Зелёный
            AntType.SCOUT: (0, 188, 212)       # Голубой
        }

        # Возможные пути к файлам картинок еды
        food_file_paths = {
            FoodType.APPLE: ["../../assets/apple.png", "assets/apple.png", "apple.png", "assets/apple.jpg"],
            FoodType.BREAD: ["../../assets/bread.png", "assets/bread.png", "bread.png", "assets/bread.jpg"],
            FoodType.NECTAR: ["../../assets/bread.png", "assets/nectar.png", "nectar.png", "assets/nectar.jpg"]
        }

        # Запасные цвета еды, если файлы не найдены
        food_fallback_colors = {
            FoodType.APPLE: (229, 57, 53),
            FoodType.BREAD: (255, 179, 0),
            FoodType.NECTAR: (255, 215, 0)
        }

        # Загрузка исходных изображений
        loaded_food_surfaces: Dict[FoodType, Optional[pygame.Surface]] = {}
        for f_type, paths in food_file_paths.items():
            surf = None
            for path in paths:
                if os.path.exists(path):
                    try:
                        surf = pygame.image.load(path).convert_alpha()
                        break
                    except Exception as e:
                        print(f"Не удалось загрузить картинку {path}: {e}")
            loaded_food_surfaces[f_type] = surf

        # Кэш масштабированных картинок еды под текущий zoom
        scaled_food_cache: Dict[Tuple[FoodType, int], pygame.Surface] = {}

        def get_scaled_food_image(f_type: FoodType, target_size: int) -> Optional[pygame.Surface]:
            if target_size <= 0:
                return None
            key = (f_type, target_size)
            if key in scaled_food_cache:
                return scaled_food_cache[key]

            orig_surf = loaded_food_surfaces.get(f_type)
            if orig_surf is not None:
                scaled = pygame.transform.smoothscale(orig_surf, (target_size, target_size))
                scaled_food_cache[key] = scaled
                return scaled
            return None

        # Точная математика odd-r (pointy-topped hexes)
        def hex_to_pixel(q: int, r: int, radius: float) -> Tuple[float, float]:
            width = math.sqrt(3) * radius
            x = width * (q + 0.5 * (r % 2))
            y = 1.5 * radius * r
            return x, y

        def get_hex_corners(cx: float, cy: float, radius: float) -> List[Tuple[float, float]]:
            corners = []
            for i in range(6):
                angle_deg = 60 * i + 30
                angle_rad = math.radians(angle_deg)
                corners.append((
                    cx + radius * math.cos(angle_rad),
                    cy + radius * math.sin(angle_rad)
                ))
            return corners

        # Распределение типов муравьёв по разным углам внутри гекса
        def get_ant_offset(ant_type: AntType, radius: float) -> Tuple[float, float]:
            if ant_type == AntType.SCOUT:
                return -0.28 * radius, 0.22 * radius   # Слева внизу
            elif ant_type == AntType.WORKER:
                return 0.0, -0.28 * radius             # Сверху по центру
            else:  # AntType.SOLDIER
                return 0.28 * radius, 0.22 * radius    # Справа внизу

        while self.is_running:
            # Берём самое актуальное состояние из очереди
            updated = False
            while not self.queue.empty():
                try:
                    game_state = self.queue.get_nowait()
                    updated = True
                except Empty:
                    break

            # Автоцентрирование на муравейнике при первом получении данных
            if updated and game_state and game_state.spot and camera_x == screen_width // 2 and camera_y == screen_height // 2:
                spot_q = game_state.spot.get('q', 0)
                spot_r = game_state.spot.get('r', 0)
                hx, hy = hex_to_pixel(spot_q, spot_r, zoom)
                camera_x = screen_width // 2 - hx
                camera_y = screen_height // 2 - hy

            mouse_pos = pygame.mouse.get_pos()

            # Обработка событий окна
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    break
                elif event.type == pygame.VIDEORESIZE:
                    screen_width, screen_height = event.w, event.h
                    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE | pygame.DOUBLEBUF)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button in (1, 3):  # ЛКМ или ПКМ для сдвига карты
                        dragging = True
                        drag_start = mouse_pos
                    elif event.button == 4:  # Масштабирование плюс
                        zoom_factor = 1.15
                        zoom = min(120.0, zoom * zoom_factor)
                        camera_x = mouse_pos[0] - (mouse_pos[0] - camera_x) * zoom_factor
                        camera_y = mouse_pos[1] - (mouse_pos[1] - camera_y) * zoom_factor
                    elif event.button == 5:  # Масштабирование минус
                        zoom_factor = 1 / 1.15
                        zoom = max(6.0, zoom * zoom_factor)
                        camera_x = mouse_pos[0] - (mouse_pos[0] - camera_x) * zoom_factor
                        camera_y = mouse_pos[1] - (mouse_pos[1] - camera_y) * zoom_factor
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (1, 3):
                        dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if dragging:
                        dx = mouse_pos[0] - drag_start[0]
                        dy = mouse_pos[1] - drag_start[1]
                        camera_x += dx
                        camera_y += dy
                        drag_start = mouse_pos
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_r):
                        if game_state and game_state.spot:
                            spot_q = game_state.spot.get('q', 0)
                            spot_r = game_state.spot.get('r', 0)
                            hx, hy = hex_to_pixel(spot_q, spot_r, zoom)
                            camera_x = screen_width // 2 - hx
                            camera_y = screen_height // 2 - hy

            # Отрисовка кадра
            screen.fill((18, 18, 18))

            hovered_hex_info = None

            if game_state:
                # 1. Отрисовка клеток карты
                for hex_item in game_state.map:
                    hx, hy = hex_to_pixel(hex_item.q, hex_item.r, zoom)
                    screen_x = hx + camera_x
                    screen_y = hy + camera_y

                    # Отсечение гексов за границами экрана
                    if screen_x < -zoom * 2 or screen_x > screen_width + zoom * 2 or \
                            screen_y < -zoom * 2 or screen_y > screen_height + zoom * 2:
                        continue

                    corners = get_hex_corners(screen_x, screen_y, zoom)
                    color = hex_colors.get(hex_item.type, (100, 100, 100))

                    # Если клетка в тумане войны - делаем её цвет затемнённым
                    if not getattr(hex_item, 'is_visible', True):
                        color = tuple(int(c * 0.5) for c in color)

                    pygame.draw.polygon(screen, color, corners)
                    pygame.draw.polygon(screen, (40, 40, 40), corners, 1)

                    # Подсветка при наведении
                    dist_sq = (mouse_pos[0] - screen_x) ** 2 + (mouse_pos[1] - screen_y) ** 2
                    if dist_sq <= (zoom * 0.85) ** 2:
                        pygame.draw.polygon(screen, (255, 255, 255), corners, 2)
                        hovered_hex_info = {
                            'q': hex_item.q, 'r': hex_item.r,
                            'type': hex_item.type.name,
                            'cost': hex_item.cost
                        }

                    # Отображение координат
                    if zoom >= 22.0:
                        lbl = font_small.render(f"{hex_item.q},{hex_item.r}", True, (30, 30, 30))
                        screen.blit(lbl, lbl.get_rect(center=(screen_x, screen_y)))

                # 2. Отрисовка еды
                for food in game_state.food:
                    hx, hy = hex_to_pixel(food.q, food.r, zoom)
                    screen_x = hx + camera_x
                    screen_y = hy + camera_y

                    if -zoom < screen_x < screen_width + zoom and -zoom < screen_y < screen_height + zoom:
                        img_size = int(zoom * 0.75)
                        food_img = get_scaled_food_image(food.type, img_size)

                        if food_img is not None:
                            # Отрисовка картинкой
                            img_rect = food_img.get_rect(center=(int(screen_x), int(screen_y)))
                            screen.blit(food_img, img_rect)
                        else:
                            # Запасная отрисовка кружком
                            f_color = food_fallback_colors.get(food.type, (255, 215, 0))
                            r_size = max(3.0, zoom * 0.28)
                            pygame.draw.circle(screen, f_color, (int(screen_x), int(screen_y)), int(r_size))
                            pygame.draw.circle(screen, (0, 0, 0), (int(screen_x), int(screen_y)), int(r_size), 1)

                        # Число ресурсов поверх/рядом
                        if zoom >= 16.0:
                            f_lbl = font_small.render(str(food.amount), True, (255, 255, 255))
                            lbl_rect = f_lbl.get_rect(center=(screen_x + zoom * 0.25, screen_y + zoom * 0.25))

                            # Подложка под цифру для читаемости
                            bg_rect = lbl_rect.inflate(4, 2)
                            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                            screen.blit(f_lbl, lbl_rect)

                # 3. Отрисовка союзных муравьёв со смещением по типам
                for ant in game_state.ants:
                    hx, hy = hex_to_pixel(ant.q, ant.r, zoom)
                    off_x, off_y = get_ant_offset(ant.type, zoom)

                    screen_x = hx + camera_x + off_x
                    screen_y = hy + camera_y + off_y

                    if -zoom < screen_x < screen_width + zoom and -zoom < screen_y < screen_height + zoom:
                        color = ant_colors.get(ant.type, (255, 255, 255))
                        ant_radius = max(3.5, zoom * 0.26)

                        pygame.draw.circle(screen, color, (int(screen_x), int(screen_y)), int(ant_radius))
                        pygame.draw.circle(screen, (255, 255, 255), (int(screen_x), int(screen_y)), int(ant_radius), 1)

                        # Индикатор HP
                        if zoom >= 18.0:
                            hp_max = getattr(ant, 'max_health', 100) or 100
                            hp_ratio = max(0.0, min(1.0, ant.health / hp_max))
                            bar_w = zoom * 0.5
                            bar_h = 3
                            bx = screen_x - bar_w / 2
                            by = screen_y - ant_radius - 4
                            pygame.draw.rect(screen, (60, 0, 0), (bx, by, bar_w, bar_h))
                            pygame.draw.rect(screen, (0, 220, 0), (bx, by, bar_w * hp_ratio, bar_h))

                # 4. Отрисовка вражеских муравьёв
                for enemy in game_state.enemies:
                    hx, hy = hex_to_pixel(enemy.q, enemy.r, zoom)
                    off_x, off_y = get_ant_offset(enemy.type, zoom)

                    screen_x = hx + camera_x + off_x
                    screen_y = hy + camera_y + off_y

                    if -zoom < screen_x < screen_width + zoom and -zoom < screen_y < screen_height + zoom:
                        ant_radius = max(3.5, zoom * 0.26)
                        pygame.draw.circle(screen, (255, 87, 34), (int(screen_x), int(screen_y)), int(ant_radius))
                        pygame.draw.circle(screen, (0, 0, 0), (int(screen_x), int(screen_y)), int(ant_radius), 1)

                # 5. Панель состояния (HUD)
                hud_bg = pygame.Surface((310, 115))
                hud_bg.set_alpha(220)
                hud_bg.fill((25, 25, 25))
                screen.blit(hud_bg, (10, 10))
                pygame.draw.rect(screen, (70, 70, 70), (10, 10, 310, 115), 1)

                txt_turn = font_large.render(f"Ход: {game_state.turn_no}", True, (255, 255, 255))
                txt_score = font_medium.render(f"Очки: {game_state.score}", True, (255, 215, 0))
                txt_ants = font_medium.render(f"Юниты: {len(game_state.ants)} | Враги: {len(game_state.enemies)}", True, (200, 200, 200))
                txt_controls = font_small.render("ЛКМ+Драг: Камера | Колёсико: Зум | Space: Центр", True, (140, 140, 140))

                screen.blit(txt_turn, (20, 16))
                screen.blit(txt_score, (20, 42))
                screen.blit(txt_ants, (20, 66))
                screen.blit(txt_controls, (20, 95))

                # 6. Инспектор клеток при наведении
                if hovered_hex_info:
                    tip_str = f"Гекс ({hovered_hex_info['q']}, {hovered_hex_info['r']}) | {hovered_hex_info['type']} | ОП: {hovered_hex_info['cost']}"
                    tip_surf = font_medium.render(tip_str, True, (255, 255, 255))
                    tw, th = tip_surf.get_width() + 16, tip_surf.get_height() + 10
                    tx, ty = mouse_pos[0] + 15, mouse_pos[1] + 15

                    if tx + tw > screen_width:
                        tx = mouse_pos[0] - tw - 10
                    if ty + th > screen_height:
                        ty = mouse_pos[1] - th - 10

                    tip_bg = pygame.Surface((tw, th))
                    tip_bg.set_alpha(230)
                    tip_bg.fill((30, 30, 30))
                    screen.blit(tip_bg, (tx, ty))
                    pygame.draw.rect(screen, (100, 100, 100), (tx, ty, tw, th), 1)
                    screen.blit(tip_surf, (tx + 8, ty + 5))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
