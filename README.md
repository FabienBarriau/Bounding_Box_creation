#Goal
The goal of this repo is to create an interface to create bounding box (for cropping or binarise) and to train a neural network that will 
replace the user when it will be enought train.

For the moment only the interface work fine, i need to change the structure of the neural network to have better result.

#How to use the interface

Click on the image to create a point.
The bounding box will be the minimal rectangle that contains all the points.
Press R "return" to erase the last point ploted.
Press Q or close the matplotlib window when you have the perfect bounding box and pass to the other image.

# Requirements

Python 3.5
* Tensorflow
* Numpy
* Matplotlib
* (argparse, os)
