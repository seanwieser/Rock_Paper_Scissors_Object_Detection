# Rock_Paper_Scissors

## Introduction

Rock Paper Scissors is a game played by two people where each player simulataneously configures their hand into the shape of a rock, paper, or scissors. Depending on what each player chose to do, a winner is determined by the following: rock beats scissors, scissors beats paper, and paper beats rock. An example of a hand in each of these configurations is shown below.

## Data
I found the dataset on Kaggle (https://www.kaggle.com/sanikamal/rock-paper-scissors-dataset?) containing pictures of different hands in each of the three configurations: rock, paper, and scissors. After some restructering and renaming of files, the directory tree is as follows:

.<br />
+-- data <br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- train<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- test<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- validation<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />

Within each paper/rock/scissors subdirectory, the image file names are of the format 'configXXX.png' where config is either paper/rock/scissors and XXX is a three digit number, including leading zeroes.

## CNN Starting Point

Image Classification with Rock Paper Scissors image dataset. Build/Use a CNN to be able to classify whether an image is of a hand in the Rock, Paper, or Scissors configuration. 

Dataset is here: https://www.kaggle.com/sanikamal/rock-paper-scissors-dataset?
Another dataset: https://www.kaggle.com/drgfreeman/rockpaperscissors/version/2?

If I am able to do this early enough in the week, I can set up my raspberry pi with a camera in order to set up a way for me to play rock, paper, scissors against my computer (with some dumb AI).
