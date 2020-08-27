import pygame
import pygame.camera
from pygame.locals import *
from time import sleep
import threading

if __name__ == "__main__":
    display_width = 200
    display_height = 300
    # pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0],(640,480))  
    cam.start()
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    idx = [0,0,0]
    idx=0
    while True:
        r = input('Press Enter to Take Picture...')
        if r != '':
            break
        image_saving = cam.get_image()
        base_path = f'../data/sean/paper/paper{idx}.png'
        pygame.image.save(image_saving, base_path)
        idx+=1
    pygame.quit()
