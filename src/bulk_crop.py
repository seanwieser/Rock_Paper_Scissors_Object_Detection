from PIL import Image
import os

if __name__ == "__main__":
    directory = '../data/sean/paper/'
    files = os.listdir(directory)
    print(files)
    for filename in files:
        im = Image.open(directory+filename)
        os.remove(directory+filename)
        region = im.crop( (0,0,200,300) )
        region.save(directory+filename)