import pygame
import pygame.camera
from pygame.font import Font
from pygame.locals import *
from time import sleep
from tensorflow.keras.models import load_model
from skimage import filters, color, io
from skimage.transform import rotate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import threading
from queue import LifoQueue

def transform_image(image):
    cropped = pygame.surfarray.array3d(image)[:200,:300]
    sobel = rotate(filters.sobel(color.rgb2gray(cropped)), -90, resize=True)
    return sobel, np.expand_dims(np.reshape(sobel, (200, 300, 1)), axis=0)

if __name__ == "__main__":
    class_labels = ['Paper', 'Rock', 'Scissors']
    model = load_model('../data/model_data/rps_model.h5')
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
    font = Font('freesansbold.ttf', 18)

    clock = pygame.time.Clock()
    while True:
        image_sur = cam.get_image()
        image_arr = transform_image(image_sur)
        probs = model.predict(image_arr[1])
        text = class_labels[np.argmax(probs)]
        textsurf = font.render(text, True, black)
        textrect = textsurf.get_rect()
        textrect.center = 40, 20
        gameDisplay.fill(white)
        gameDisplay.blit(image_sur, (0,0))
        gameDisplay.blit(textsurf, textrect)
        pygame.display.update()
        clock.tick(60)
        model.reset_states()

    



