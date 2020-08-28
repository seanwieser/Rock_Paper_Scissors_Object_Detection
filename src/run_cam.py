import pygame
import pygame.camera
from pygame.locals import *
from time import sleep
from tensorflow.keras.models import load_model
from skimage import filters, color, io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import threading
from queue import LifoQueue

def transform_image(image):
    cropped = pygame.surfarray.array3d(image)[:200,:300]
    sobel = filters.sobel(color.rgb2gray(cropped))
    return np.expand_dims(np.reshape(sobel.astype('float') / 255, (200, 300, 1)), axis=0)

def background():
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
        q.put(image)
        if q.qsize() > 1000:
            q.queue.clear()
        gameDisplay.fill(white)
        gameDisplay.blit(image, (0,0))
        pygame.display.update()
        clock.tick(60)

# def foreground():



if __name__ == "__main__":
    flag = False
    q = LifoQueue()
    model = load_model('../data/model_data/rps_model.h5')

    b = threading.Thread(name='background', target=background)
    # f = threading.Thread(name='foreground', target=foreground)

    b.start()
    while True:
        input('Press Enter to predict capture...')
        image_sur = q.get()
        image_arr = transform_image(image_sur)
        probs = model.predict(image_arr)
        model.reset_states()
        print(probs)
    # f.start()
    



