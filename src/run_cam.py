# import pygame
# import pygame.camera
# from pygame.locals import *
from time import sleep
from tensorflow.keras.models import load_model
from skimage import filters, color, io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np



if __name__ == "__main__":
    display_width = 200
    display_height = 300
    # pygame.init()
    # pygame.camera.init()
    # camlist = pygame.camera.list_cameras()
    # if camlist:
    #     cam = pygame.camera.Camera(camlist[0],(640,480))  
    # cam.start()
    # gameDisplay = pygame.display.set_mode((display_width,display_height))
    sean_data_dir = '../data/temp'
    sean_test_datagen = ImageDataGenerator(rescale=1. / 255)
    sean_generator = sean_test_datagen.flow_from_directory(
        sean_data_dir,
        target_size=(200, 300),
        batch_size=16,
        class_mode='categorical')
    predict_images = np.reshape(io.imread_collection('../data/sean_test/paper/*.png').astype('float') / 255, (200, 300, 1))
    predict_images = np.expand_dims(predict_image, axis=0)
    model = load_model('../data/model_data/rps_model.h5')
    print('Predicting', predict_images.shape)
    # print(model.summary())
    for image in predict_images:
        print(model.predict(predict_image, verbose=1))
    
    
    
    # black = (0,0,0)
    # white = (255,255,255)
    # clock = pygame.time.Clock()
    # while True:
    #     image = cam.get_image()
    #     image_arr = pygame.surfarray.array3d(image)[:display_width,:display_height]
    #     prediction = model.predict(filters.sobel(color.rgb2gray(image_arr)))
    #     print(prediction)
    #     # print(type(image_arr), image_arr.shape)
    #     gameDisplay.fill(white)
    #     gameDisplay.blit(image, (0,0))
    #     pygame.display.update()
    #     clock.tick(60)


