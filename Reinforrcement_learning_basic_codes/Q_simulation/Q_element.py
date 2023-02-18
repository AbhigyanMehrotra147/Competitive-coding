import pygame
import numpy as np
import common_methods as cm
from Q_element_data import Q_element_data

image_url = "element_background.jpg"
font_style = ""
# Class handles a single element in the Q_table


class Q_element:

    """
    1. self._super_surface
    2. image_url
    3. font_style
    4. relative_size  is the size of self._screen relative to the size of the screen passed as an argument in the constructor
    5. relative_coords are the coordinates of self._screen relative to the size of the screen passed as an argumnet

    """

    def __init__(self, screen: pygame.Surface, relative_size, relative_coords, color: tuple = (255, 255, 250, 1), data=0):
        self._super_surface = screen

        self._rect = None

        self._relative_size = relative_size
        self._relative_coords = relative_coords
        self._size = 0
        self._coords = 0

        self._color = color

        self._data = data

        self._create()

    def _create_surface(self):

        self._size = cm.get_relative_size(
            relative_surface=self._super_surface, relative_size=self._relative_size)
        self._coords = cm.get_relative_coords(
            relative_surface=self._super_surface, relative_coords=self._relative_coords)
        self._surface = pygame.Surface(size=self._size)
        self._surface.set_colorkey(self._color)
        self._surface.fill(color=self._color)

        self._rect = pygame.Rect(self._coords, self._size)

        # print(self._size)

    def _create(self):
        self._create_surface()
        self._data = Q_element_data(
            super_surface=self._surface, data=self._data)

    def _render_self(self):

        self._super_surface.blit(source=self._surface, dest=self._rect)

    def render(self):
        self._data.render()
        self._render_self()
