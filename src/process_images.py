import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, filters
from skimage.transform import resize, rotate

def rotate_images(images, kind):
    print(f'Tranforming: {kind}')
    for idx, image in enumerate(images):
        io.imsave(f'../rps-cv-images/{kind}_processed/{kind}{str(idx)}.png', rotate(image, -90, resize=True))

def is_white(rgb):
    return (rgb[0]>100 and rgb[1]>100 and rgb[2]>100)
    
def print_image_arrays(images):
    for image in images:
        print(image[150][100])

def save_sobels(images, kind):
    print(f'Tranforming: {kind}')
    for idx, image in enumerate(images):
        sobel_img = filters.sobel(color.rgb2gray(image))
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         if is_white(image[i][j]):
        #             image[i][j] = new_rgb
        io.imsave(f'../data/train/{kind}_processed/{kind}{str(idx)}.png', sobel_img)

if __name__ == "__main__":
    paper_paths = '../data/train/paper/*.png'
    paper_sobels_path = '../data/train/paper/*.png'
    rock_paths = '../data/train/rock/*.png'
    scissor_paths = '../data/train/scissors/*.png'
    paper_images = io.imread_collection(paper_paths)
    paper_sobels = io.imread_collection(paper_sobels_path)
    rock_images = io.imread_collection(rock_paths)
    scissor_images = io.imread_collection(scissor_paths)
    # rotate_images(paper_images, 'paper')
    # rotate_images(rock_images, 'rock')
    # rotate_images(scissor_images, 'scissors')
    # save_sobels(rock_images, 'rock')
    # save_sobels(paper_images, 'paper')
    # save_sobels(scissor_images, 'scissors')
    print_image_arrays(paper_sobels)