import pygame
import common_methods as cm
import numpy as np
font_style = "freesansbold.ttf"
# Class handles a single element in the Q_table


class Q_element_font:

    """
    """
    # Class does not need relative coords as the relative coords are taken out

    def __init__(self, super_surface: pygame.Surface, relative_size: np.float = 0.5, color=(200, 200, 200, 1),
                 background_colour=(0, 0, 0, 1), data: float = 0,
                 font_style: str = font_style):
        self._super_surface = super_surface
        self._surface = None
        self._font = None
        self._text = None
        self._text_rect = None

        self._font_style = font_style
        self._data = round(data, 2)
        self._data = str(self._data)

        self._relative_size = relative_size
        self._size = None
        self._coords = None
        self._color = color
        self._background_color = background_colour

        self._create()

    def _create_font(self):
        self._size = self._relative_size*self._super_surface.get_size()[0]
        self._font = pygame.font.Font(self._font_style, int(self._size))

    def _create_text(self):
        self._text = self._font.render(
            self._data, True, self._color, self._background_color)

    def _create_text_rect(self):
        self._text_rect = self._text.get_rect()
        super_surface_size = self._super_surface.get_size()
        self._text_rect.center = (
            super_surface_size[0]//2, super_surface_size[1]//2)

    def _create(self):
        self._create_font()
        self._create_text()
        self._create_text_rect()

    def _render_text(self):
        self._super_surface.blit(source=self._text, dest=self._text_rect)

    def _render_self(self):
        self._super_surface.blit(source=self._surface, dest=self._rect)

    def render(self):
        self._render_text()
        # self._render_self()
