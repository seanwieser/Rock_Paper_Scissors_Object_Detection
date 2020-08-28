# Rock_Paper_Scissors

## Introduction

Rock Paper Scissors is a game played by two people where each player simulataneously configures their hand into the shape of a rock, paper, or scissors. Depending on what each player chose to do, a winner is determined by the following: rock beats scissors, scissors beats paper, and paper beats rock. An example of a hand in each of these configurations is shown below.

## Dataset Journey
### Inital Training Set

I found the dataset on Kaggle (https://www.kaggle.com/drgfreeman/rockpaperscissors/version/2?) containing pictures of different hands in each of the three configurations: rock, paper, and scissors. Each one of these configurations (or classes) had a little over 700 images each, which seemed like enough for simple image classification. An example of images from this set are shown below:

green unrotated dataset

### Adding My Own Images

I trained an initial model that performed well with validation data that came from the kaggle dataset. That was encouraging but not that exciting to me because I have no idea whether or not it will be able to predict images of my own hand! I achieved this task by writing some utility scripts to produce images that mimic the images in the kaggle dataset. 

- bulk_crop.py - Crops and resaves all images in a directory to 200x300 pixels

- lap_cam.py - Captures and saves images using my laptop camera. Captures are made everytime the user clicks 'Enter'

- process_images.py - rotate: Rotates and resaves all images in directory 90 degrees clockwise
                      save_sobels: Saves sobel filtered images of all images in a directory



After some restructering and renaming of files, the directory tree is shown below:

.<br />
+-- data <br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- train<br /> (654 files each)
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- val<br /> (122 files each)
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- test<br /> (42 files each)
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />




### Preprocessing


## CNN Architecture
The architecture for my CNN was inspired by the blog post https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

## Using AWS Instance



## Model Performance

## Using CNN in Realtime

## Conclusion

## Next Steps
