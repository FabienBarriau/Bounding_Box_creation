from PIL import Image
import matplotlib.pyplot as plt
from utils import create_model, BoundingBoxDraw
import numpy as np
import os
from tensorflow import keras
from numpy.linalg import inv
import pandas as pd

"""
    reference(to del once add to readme) :
     PIL: https://pillow.readthedocs.io/en/3.1.x/index.html
"""

# LOCATION OF THE IMAGE
FILE = r"C:\Users\Mojun\python_project\Bounding_Box_creation\Image"
# Number of images to crop before training ie to feed the neural network
BATCH = 1
# Number of batch to do before showing a suggested BB
SHOW_RESULT = 1
# max number of images, should be the max number of images you want to feed/crop
NBR_MAX = 100

"""
    Use VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
        Very Deep Convolutional Networks for Large-Scale Image Recognition
        K. Simonyan, A. Zisserman
        arXiv:1409.1556
    No particular reason to use it. The result look ok for a demo
    We only use the CNN part of it to recognize our features
"""
VGG16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                       input_shape=(224, 224, 3), pooling="max", classes=1000)

"""
    Create the second neural network.
    This NN will be train to create meaningful bounding box (BB)
"""
BoundingBoxNet = create_model()

# get the name of all the image to use
names_list = [name for name in os.listdir(FILE) if name.endswith(".jpg") or name.endswith(".png")]

"""
    init some basic parameters:
        shape_list: Useless for now: unused
        encoding_list_temp: 
        bb_list_temp: 
        bb_list: 
"""
shape_list = list()
encoding_list_temp = list()
bb_list_temp = list()
bb_list = list()

# for all images
# We work now image by image
for c, name in enumerate(names_list):

    # append \ to end of FILE if necessary
    if not FILE[-1] == "\\":
        FILE += "\\"

    # load image and use RGB encoding to work with
    im = Image.open(FILE + name).convert("RGB")

    """
        scaling = transformation from square [224, 224] to rectangle im.size
        
        VGG16 work on 224 by 224 image size.
        
        Touch only the Bounding Box:
        scaling: determine how to reduce our BB size to fit in [224, 224] image
        inv_scaling: determine how to augment the predicted BB on [224, 224] image to fit the original image size
    """
    shape = np.array([im.size[0], im.size[1]])
    shape_list.append(shape)
    scaling = np.diag(shape) / 224
    inv_scaling = inv(scaling)

    """
        Encoding at last convolutional layer of VGG16.
        VGG16 identified the useful features in our image, so that when our second NN to predict a BB it has
        meaningful feature to work on, instead of pixel/localisation shit
        cf classical CNN explication 
    """
    mini_im_tensor = np.expand_dims(np.asarray(im.resize([224, 224], Image.NEAREST)) / 255, axis=0)
    encoding = VGG16.predict(mini_im_tensor, batch_size=BATCH, verbose=1)
    encoding_list_temp.append(encoding)


    # BoundingBoxNet predict the value for the 2 points who define the bounding box.
    pred = np.reshape(BoundingBoxNet.predict(np.expand_dims(encoding, axis=0)), [2, 2]) * 224
    """
        Sometimes the predicted BB can be outside our image, so we reshape to fit in.
        The true bounding box is limited by the square [[0,0],[224, 224]]
    """
    pred_limited = np.maximum(np.minimum(pred, np.array([[224, 224], [224, 224]])), np.array([[0, 0], [0, 0]]))
    # Apply the inverse scaling to retrieve the bounding box of the inital space.
    pred_metrics = pred_limited.dot(scaling)

    # Create the figure and show it
    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    ax.imshow(im)
    # fit the size of the window
    plt.get_current_fig_manager().window.state('zoomed')

    # Show the predicted BB only when asked ie when the parameter ask for it
    # enough train -> for now it determine purely by the parameter at the beginning
    if c < SHOW_RESULT * BATCH:
        # Call the app with nothing as initial bounding box when the network is not enough train.
        bb_app = BoundingBoxDraw(ax, fig)
        plt.show(fig)
    else:
        # Call the app with the bb predict by BoundingBoxNet as the initial bounding box
        # when the network is enough train.
        bb_app = BoundingBoxDraw(ax, fig, pred_metrics)
        plt.show(fig)
    """
        Stock bounding box in a list (and resize it to fit the original image if inv_scaling is really that) 
        to save it later.
    """
    bb = bb_app.get_bounding_box(format="ul_br")
    bb_list_temp.append(np.reshape(bb.dot(inv_scaling), [1, 4]) / 224)
    bb_list.append(np.reshape(bb.dot(inv_scaling), [1, 4]) / 224)

    # Learn when there are enough data (enough = BATCH) ie the learning happened here
    # enough data -> for now it determine purely by the parameter at the beginning
    if c % BATCH == 0:
        loss = BoundingBoxNet.fit(x=np.stack(encoding_list_temp, axis=0), y=np.stack(bb_list_temp, axis=0),
                                  validation_split=0., batch_size=BATCH, epochs=1)
        encoding_list_temp.clear()
        bb_list_temp.clear()
        count = 0

    """  
        Stop the training at max number of iteration.
        Useful to try/train your model on a sample to see the result faster.
    """
    if c == (NBR_MAX - 1):
        break

    print(c)

# Write the predicted BB in a file
df = pd.DataFrame(np.vstack(bb_list), columns=["ul_x", "ul_y", "dr_x", "dr_y"])
df.to_csv(r"C:\Users\Mojun\python_project\Bounding_Box_creation\df.csv", sep=";")
