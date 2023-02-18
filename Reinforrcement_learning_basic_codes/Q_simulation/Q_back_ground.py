import pygame
from Q_element import Q_element
import common_methods as cm
# Some class members for the class Q_background

image_url = "./matrix_background_better.jpg"
No_rows = 16
No_cols = 4

# class acts as the background to the Q_table
# All the elements of the Q_table will be blitted on this itself


class Q_Background:
    """
    1. self._screen is the matrix
    2.
    3.
    4.
    5.
    6.
    7.
    8.
    9.
    10.
    """

    def __init__(self, super_screen: pygame.Surface, destination: tuple = [0, 0], No_rows: int = No_rows, No_cols: int = No_cols,
                 image_url: str = image_url, relative_size: tuple = (0.9, 0.7), color=[0, 0, 0, 1]):

        self._super_surface = super_screen
        self._screen = None

        self._size = relative_size
        self._coords = destination

        self._No_rows = No_rows
        self._No_cols = No_cols
        self._No_elements = self._No_cols*self._No_rows
        self._element_matrix = []

        self._color = color
        self._create(screen=super_screen)

    def _create_self(self, screen: pygame.surface):

        self._coords = [0 + 0.5*(1-self._size[0]), 0 + 0.5*(1-self._size[1])]
        self._coords = cm.get_relative_coords(
            relative_surface=self._super_surface, relative_coords=self._coords)

        self._size = cm.get_relative_size(
            relative_surface=self._super_surface, relative_size=self._size)

        self._screen = pygame.Surface(self._size)
        self._screen.fill(self._color)
        self._screen.set_colorkey(self._color)

    def _render_elements(self, F_L_table):

        relative_coords = [0, 0]
        relative_size = (1/self._No_rows, 1/self._No_cols)
        # print(relative_coords)
        # print(relative_size)
        for i in range(self._No_rows):
            # print(relative_coords)
            for j in range(self._No_cols):
                Q = Q_element(
                    screen=self._screen, relative_coords=(relative_coords[0], relative_coords[1]), relative_size=relative_size, data=F_L_table[i, j])
                Q.render()
                # print(1/self._No_cols)
                relative_coords[1] += (relative_size[1])
            relative_coords[1] = 0
            relative_coords[0] += relative_size[0]
        # Q = Q_element(screen=self._screen, relative_coords=(0, 0),
        #               relative_size=(0.2, 0.2))
        # Q.render()

    def _update_color(self):
        if(self._color[0] > 255):
            self._color = 0
        else:
            self._color[0] += 1
        self._screen.fill(self._color)

    def _create(self, screen: pygame.surface):
        self._create_self(screen=screen)
    # function renders self and all imported classes

    def render(self, F_L_table):
        # self._update_color()
        self._render_elements(F_L_table)
        self._render_self()

    def _render_self(self):
        self._super_surface.blit(source=self._screen, dest=self._coords)
