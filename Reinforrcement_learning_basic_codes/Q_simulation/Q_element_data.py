import pygame
import common_methods as cm
from Q_element_font import Q_element_font
font_style = ""
# Class handles a single element in the Q_table


class Q_element_data:

    """
    """
    # Class does not need relative coords as the relative coords are taken out

    def __init__(self, super_surface: pygame.Surface, relative_size: float = (0.9, 0.9), color=(176, 224, 230, 1), data=0):
        self._super_surface = super_surface
        self._surface = None
        self._rect = None

        self._relative_size = relative_size
        self._color = color

        self._font_n_data = None

        self._data = data

        self._create()

    def _create_surface(self):

        self._coords = (
            (1-self._relative_size[0])*0.5, (1 - self._relative_size[1])*0.5)
        self._size = cm.get_relative_size(
            relative_surface=self._super_surface, relative_size=self._relative_size)

        self._coords = cm.get_relative_coords(
            relative_surface=self._super_surface, relative_coords=self._coords)
        self._surface = pygame.Surface(size=self._size)
        self._surface.fill(color=self._color)
        self._surface.set_colorkey(self._color)
        self._rect = pygame.Rect(self._coords, self._size)
        # print(self._size)

    def _create(self):
        self._create_surface()
        self._font_n_data = Q_element_font(
            super_surface=self._surface, data=self._data)

    def _render_self(self):
        self._super_surface.blit(source=self._surface, dest=self._rect)

    def render(self):
        self._font_n_data.render()
        self._render_self()
