from __future__ import annotations

##########
#  minigrid 註解版（增量）
##########
# 這份檔案只保留「目前正在研究」的段落。
# 研究策略：先掌握 full render -> 再掌握 grid 的實際繪圖。

import math
from typing import Any

import numpy as np
import pygame
import pygame.freetype
from minigrid.core.constants import COLORS, TILE_PIXELS
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


class MiniGridEnv:
    ##########
    #  FULL RENDER（全地圖）
    ##########
    # 這段的任務不是「畫格子」，而是「決定哪些格子要高亮」。
    # 真正的格子繪製在 Grid.render / Grid.render_tile。
    def get_full_render(self, highlight, tile_size):
        # 取得 agent 視角下的可見遮罩（局部）
        _, vis_mask = self.gen_obs_grid()

        # 方向向量：前進方向與右方向
        f_vec = self.dir_vec
        r_vec = self.right_vec

        # 以目前朝向推回視野左上角（世界座標）
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # 全地圖高亮遮罩（預設不高亮）
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # 掃過局部視野中的每個格子
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # 局部不可見就跳過
                if not vis_mask[vis_i, vis_j]:
                    continue

                # 局部座標 -> 世界座標
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                # 越界保護
                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # 可見格設為高亮
                highlight_mask[abs_i, abs_j] = True

        # 把高亮遮罩交給 grid 真正繪製
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    ##########
    #  RENDER（顯示包裝層）
    ##########
    # 這段不負責畫格子內容，只負責：
    # 1) 取得 get_frame 的結果
    # 2) 轉成 pygame 可顯示格式
    # 3) 疊背景與 mission 文字
    # 4) blit + flip 真正顯示到視窗
    def render(self):
        # 先拿到目前 frame（內容來源是 get_frame -> Grid.render）
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            # numpy 軸轉換成 pygame.surfarray 常用格式
            img = np.transpose(img, axes=(1, 0, 2))

            # 第一次 render 記錄尺寸
            if self.render_size is None:
                self.render_size = img.shape[:2]

            # 第一次 render 才初始化視窗
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")

            # 第一次 render 才建立 FPS clock
            if self.clock is None:
                self.clock = pygame.time.Clock()

            # numpy -> Surface
            surf = pygame.surfarray.make_surface(img)

            # 建立背景並預留 mission 文字空間
            offset = surf.get_size()[0] * 0.1
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            # 縮放到視窗尺寸
            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            # mission 文字
            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            # 貼到視窗 + 更新事件 + 限幀 + 刷新
            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img


class Grid:
    # 與原檔一致：render_tile 使用的靜態快取
    tile_cache: dict[tuple[Any, ...], Any] = {}

    ##########
    #  單格繪製：render_tile
    ##########
    # 這裡負責「一個 cell」的像素內容。
    # 順序：格線 -> 物件 -> agent三角形 -> 高亮 -> 抗鋸齒 -> cache。
    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
    ) -> np.ndarray:
        # cache key：相同物件+方向+高亮+尺寸可重用
        key: tuple[Any, ...] = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # 格線（上/左）
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # 物件本體（wall/goal/door/...）
        # obj.render(img) = 讓該物件自己把外觀畫到當前 tile 的像素緩衝區 img 上。
        # 例如 Wall 會畫滿灰色方塊、Goal 會畫綠色方塊、Door 會依開關狀態畫不同圖形。
        # 這一步完成後，tile 才有「物件外觀」可被後續 Grid.render 拼回整張地圖。
        if obj is not None:
            obj.render(img)

        # agent 三角形覆蓋層
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # 視野高亮
        if highlight:
            # highlight_img(img) 會在當前 tile 疊一層亮化效果（偏白/提亮）。
            # 這不是改物件類型，而是純視覺提示：告訴使用者這格在可見範圍內。
            # 在 full render 中，是否 highlight 由 get_full_render 算出的 highlight_mask 決定。
            highlight_img(img)

        # 抗鋸齒：先高解析繪製，再降採樣
        img = downsample(img, subdivs)

        # 寫入 cache
        cls.tile_cache[key] = img

        return img


    ##########
    #  全圖繪製：render
    ##########
    # 這裡把每個 tile 拼回整張地圖。
    def render(
        self,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # 逐格繪製 + 貼回大圖
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img  # tile 貼圖

        return img
