# Rock_Paper_Scissors

![alt text](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/title.jpg "")

## Introduction

Rock Paper Scissors is a game played by two people where each player simulataneously configures their hand into the shape of a rock, paper, or scissors. Depending on what each player chose to do, a winner is determined by the following: rock beats scissors, scissors beats paper, and paper beats rock.

As lonely as it sounds, I love the idea of creating a way to play the game in a natural way against an unnatural opponent, my computer! The only thing stopping us from doing this already is that the computer can't 'see' what you chose to play on a given round. In order to solve this problem, I decided to build an image classifier by training a convolutional neural network on images of hands.

## Dataset Journey
### Inital Training Set

I found the dataset on Kaggle (https://www.kaggle.com/drgfreeman/rockpaperscissors/version/2?) containing pictures of different hands in each of the three configurations: rock, paper, and scissors. Each one of these configurations (or classes) had a little over 700 images each, which seemed to be enough for simple image classification. An example of images from this set are shown below:

| ![rock](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_rock.png)  | ![paper](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_paper.png) | ![scissors](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_scissors.png)

Each image is 300x200 pixels. The images as a whole have consistent background and lighting but have many styles of hands and different ways they make up each of the three configurations. I was happy to see there was consistency in all the parts of the image that I wasn't predicting on.

### Adding My Own Images

I trained an initial model that performed well with validation data that came from the kaggle dataset. That was encouraging but not that exciting to me because I have no idea whether or not it will be able to predict images of my own hand! I achieved this task by writing some utility scripts to produce images that mimic the images in the kaggle dataset. 

- lap_cam.py - Captures and saves images using my laptop camera. Captures are made everytime the user clicks 'Enter'

- bulk_crop.py - Crops and resaves all images in a directory to 200x300 pixels in the top left of source image

- process_images.py - rotate: Rotates and resaves all images in directory 90 degrees clockwise. Used for Kaggle images.
                      save_sobels: Saves sobel filtered images of all images in a directory. Used on all images.

| ![rock](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_rock.png)  | ![paper](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_paper.png) | ![scissors](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_scissors.png)

### Final Dataset

| ![rock](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_rock_sobel.png)  | ![paper](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_paper_sobel.png) | ![scissors](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/kaggle_scissors_sobel.png)

| ![rock](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_rock_sobel.png)  | ![paper](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_paper_sobel.png) | ![scissors](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/sean_scissors_sobel.png)

In order to use the incredible functionality of the ImageDataGenerator class (described below), the final structure of the directory tree is as follows:

.<br />
+-- data <br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- train (654 files each)<br /> 
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- val (122 files each)<br /> 
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />
|&nbsp;&nbsp;&nbsp;&nbsp;+-- test (42 files each)<br /> 
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- paper<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- rock<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;+-- scissors<br />

### Preprocessing

Even though I have my image dataset polished off, I don't want to directly feed them into a model to train. In order to get more variety of images to train on, I decided to use the ImageDataGenerator class in keras. This produces many augmentations of all the images for the model to train on. Hopefully this will help the model be more robust in terms of inputs it can predict well on.

It might help to look at a snippet of code to see how I implemented this class:


```
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
    sean_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True)

    validation_generator = sean_test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False)
```

It is important to think about the images that are being used. First, the hand in each image is oriented and sized in a fairly consistent way. This means that I don't need to augment the images in any extreme way to be able to predict on inputs that are similar to the images that the model is training on. If I expected to have inputs that are varying to a higher degree, I might have been more aggressive with the data augmentation and preproccessing.   

## CNN Architecture
The architecture for my CNN is very simple.

```
model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
```

## Using AWS Instance
Maybe not quickly enough, it became apparent to me that my laptop is not capable of running tensorflow in any productive way. In order to overcome this obstacle, I decided to spin up an Amazon Web Services instance. The instance types can be found here: https://aws.amazon.com/ec2/instance-types/

I decided on the 'p2.xlarge' instance type under the Accelerated Computing section because it has a dedicated NVIDIA GPU. Using this instance to train a model with tensorflow made this project viable but it did take substantial time to set up in the correct way. Some of the set up included installing tensorflow-gpu, setting up NVIDIA drivers, ensuring version compatibility, and copying over image data. Each of these had to be done several times throughout the week as my project developed.

I communicated to the AWS instance through SSH and GitHub. Image data and model data were transferred through SSH while code changes were transferred through GitHub. Because of the CLI aspect of SSH and my fear of code becoming out of sync, I always made changes to the code on my local computer then went through a push/pull cycle.

## Model Performance
Training over 100 Epochs
![training](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/accuracy_100_epochs.png "Training over 100 Epochs")

Training over 30 Epochs
![training](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/accuracy.png "Training over 30 epochs")

Validation Confusion Matrix
![cm](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/val_cm.png "Validation Confusion Matrix")

Test Confusion Matrix
![cm](https://github.com/seanwieser/Rock_Paper_Scissors/blob/master/images/test_cm.png "Test Confusion Matrix")


## Using CNN in Realtime

## Conclusion

## Next Steps
