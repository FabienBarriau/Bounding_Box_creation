from PIL import Image
import matplotlib.pyplot as plt
import utils as gui
import numpy as np
import os
from tensorflow import keras
from numpy.linalg import inv
import pandas as pd

FILE = "C:/Users/fabien/Desktop/Etheostoma/Google Doc/2013/BAI/Original/"
BATCH = 5
SHOW_RESULT = 15
NBR_MAX = 100

VGG16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                       input_shape=(224, 224, 3), pooling="max", classes=1000)

BoundingBoxNet = gui.create_model()

names_list = list()
for name in os.listdir(FILE):
    extension = name.split(".")[1]
    if extension.lower() in ["jpg", "png"]:
        names_list.append(name)

shape_list = list()
encoding_list_temp = list()
bb_list_temp = list()
bb_list = list()

for c, name in enumerate(names_list):

    im = Image.open(FILE + name).convert("RGB")

    # scaling = transformation from square [224, 224] to rectangle im.size
    shape = np.array([im.size[0], im.size[1]])
    shape_list.append(shape)
    scaling = np.diag(shape) / 224
    inv_scaling = inv(scaling)

    # Encoding at last convolutional layer of VGG16
    mini_im_tensor = np.expand_dims(np.asarray(im.resize([224, 224], Image.NEAREST))/255, axis=0)
    encoding = VGG16.predict(mini_im_tensor, batch_size=BATCH, verbose=1)
    encoding_list_temp.append(encoding)

    # BoundingBoxNet predict the value for the 2 points who define the bounding box.
    pred = np.reshape(BoundingBoxNet.predict(np.expand_dims(encoding, axis=0)), [2, 2])*224
    # The true bounding box is limited by the square [[0,0],[224, 224]]
    pred_limited = np.maximum(np.minimum(pred, np.array([[224, 224], [224, 224]])), np.array([[0, 0], [0, 0]]))
    # Apply the inverse scaling to retrieve the bounding box of the inital space.
    pred_metrics = pred_limited.dot(scaling)

    # Create the figure
    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    ax.imshow(im)
    plt.get_current_fig_manager().window.state('zoomed')

    if c < SHOW_RESULT*BATCH:
        # Call the app with nothing as initial bounding box when the network is not enough train.
        bb_app = gui.BoundingBoxDraw_app(ax, fig)
        plt.show(fig)
    else:
        # Call the app with the bb predict by BoundingBoxNet as the initial bounding box
        # when the network is enough train.
        bb_app = gui.BoundingBoxDraw_app(ax, fig, pred_metrics)
        plt.show(fig)

    # Stock bounding box in a list
    bb = bb_app.get_bounding_box(format="ul_br")
    bb_list_temp.append(np.reshape(bb.dot(inv_scaling), [1, 4])/224)
    bb_list.append(np.reshape(bb.dot(inv_scaling), [1, 4])/224)

    # Learn when there are enough data (enough = BATCH)
    if c % BATCH == 0:
        loss = BoundingBoxNet.fit(x=np.stack(encoding_list_temp, axis=0), y=np.stack(bb_list_temp, axis=0),
                                  validation_split=0., batch_size=BATCH, epochs=1)
        encoding_list_temp.clear()
        bb_list_temp.clear()
        count = 0

    if c == (NBR_MAX-1):
        break

    print(c)

df = pd.DataFrame(np.vstack(bb_list), columns=["ul_x", "ul_y", "dr_x", "dr_y"])
df.to_csv("C:/Users/fabien/PycharmProjects/BoundingBoxDirectLearning/df.csv", sep=";")

