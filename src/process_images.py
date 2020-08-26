import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, filters
from skimage.transform import resize, rotate

def rotate_images(images, kind):
    print(f'Tranforming: {kind}')
    for idx, image in enumerate(images):
        io.imsave(f'../rps-cv-images/{kind}_processed/{kind}{str(idx)}.png', rotate(image, -90, resize=True))

if __name__ == "__main__":
    paper_paths = '../rps-cv-images/paper/*.png'
    rock_paths = '../rps-cv-images/rock/*.png'
    scissor_paths = '../rps-cv-images/scissors/*.png'
    paper_images = io.imread_collection(paper_paths)
    rock_images = io.imread_collection(rock_paths)
    scissor_images = io.imread_collection(scissor_paths)
    rotate_images(paper_images, 'paper')
    rotate_images(rock_images, 'rock')
    rotate_images(scissor_images, 'scissors')