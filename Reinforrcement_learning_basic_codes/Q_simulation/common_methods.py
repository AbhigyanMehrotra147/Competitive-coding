import pygame


def get_relative_coords(relative_surface: pygame.Surface, relative_coords: tuple) -> list:
    surf_size = relative_surface.get_size()
    return [int(surf_size[0]*relative_coords[0]), int(surf_size[1]*relative_coords[1])]


def get_relative_size(relative_surface: pygame.Surface, relative_size: tuple) -> list:
    surf_size = relative_surface.get_size()
    return [int(surf_size[0]*relative_size[0]), int(surf_size[1]*relative_size[1])]
