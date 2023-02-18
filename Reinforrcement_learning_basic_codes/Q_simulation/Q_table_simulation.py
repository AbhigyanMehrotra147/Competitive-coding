import pygame
import numpy as np
from Frozen_lake_formal import F_L


class Q_simulation:

    image_url = "./matrix_background_better.jpg"
    """
    1.
    2.
    3.
    4.
    5.
    6.
    7.
    8.
    9.
    10.
    11.
    12.
    13.

    """
    #

    def __init__(self, Q_tables: np.array, window_size: tuple, title: str, window_color: tuple, screen_destination: tuple, image_url=image_url):
        self.Q_tables = Q_tables

        self._window = None
        self._screen = None
        self._window_size = window_size
        self._window_color = window_color
        self._flags = pygame.RESIZABLE

        self._screen_destination = screen_destination

        self._image_url = image_url

        self._title = title

        self._running = True

        self._F_L_table = F_L._Q_table_over_time
    # Function initializes all of the window's characterstics

    def _initialize(self):
        pygame.init()
        if(not pygame.get_init()):
            print("Error initializing")
            exit(code=-1)
        else:
            from Q_back_ground import Q_Background
            self._window = pygame.display.set_mode(
                size=self._window_size, flags=self._flags, depth=1, display=0, vsync=0)
            # Remember that action takes no keyword arguments
            pygame.display.set_caption(self._title)
            self._screen = pygame.image.load(
                self._image_url).convert()

            self._matrix_Background = Q_Background(
                super_screen=self._screen, destination=self._screen_destination)

    # Function handles only the events
    def _event_handling(self):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                self._running = False
            elif(event.type == pygame.VIDEORESIZE):
                self._window_size = event.dict["size"]

    def _update():
        pass

    # Functions renders only the self
    def _render_self(self):

        self._window.fill(color=self._window_color)

        self._window.blit(pygame.transform.scale(
            self._screen, self._window_size), dest=self._screen_destination)

    # Function renders self and all imported classes

    def _render(self, index):
        self._matrix_Background.render(self._F_L_table[index])
        self._render_self()
        pygame.display.update()

    # Function exeutes the simulation
    def execute(self):
        self._initialize()
        index = 0
        while(self._running and index < self._F_L_table.shape[0]):
            self._event_handling()
            self._render(index=index)
            index += 1
        self._clean()
        print("Program game finished")

    #
    def _clean(self):
        pygame.quit()


Q_tables = None

# Window specifications
window_size = (800, 600)
window_color = (0, 0, 0)

# screen_specifications
screen_destination = (0, 0)

title = "Q_table simulation"


Q_s = Q_simulation(Q_tables=Q_tables, window_size=window_size, title=title,
                   window_color=window_color, screen_destination=screen_destination)
Q_s.execute()
