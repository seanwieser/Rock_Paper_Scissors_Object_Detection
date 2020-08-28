import pygame
import pygame.camera
from pygame.locals import *
from time import sleep


if __name__ == "__main__":
    display_width = 200
    display_height = 300
    pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0],(640,480))  
    cam.start()
    gameDisplay = pygame.display.set_mode((display_width,display_height))

    black = (0,0,0)
    white = (255,255,255)
    clock = pygame.time.Clock()
    while True:
        image = cam.get_image()
        image_arr = pygame.surfarray.array3d(image)[:display_width,:display_height]
        print(type(image_arr), image_arr.shape)
        gameDisplay.fill(white)
        gameDisplay.blit(image, (0,0))
        pygame.display.update()
        clock.tick(60)


