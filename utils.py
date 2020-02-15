import matplotlib.patches as patches
from tensorflow import keras
import numpy as np
import tensorflow as tf


# Create the shape and compile the model for our NN (NOT VGG)
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(units=84, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(units=4, activation=tf.sigmoid)
    ])
    # TODO: make hyperparameters to choose all of this
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])

    return model


class BoundingBoxDraw:

    def __init__(self, ax, fig, bounding_box_init=None):

        self.ax = ax

        """
            A bounding box is always define by only 2 points (xmin/ymin and xmax/ymax) since its a rectangle.
            So we only need 2 point on our image to determine/draw the BB.
            
            self.points_list represent the value of the two extra points needed to define our rectangle.
            self.bounding_box represent literally the BB we use, it can be compress in two points(np.array([[0, 0], [0, 0])
            but we use 4 for ease of calculus
        """

        # Happen when the model is not train enough
        if bounding_box_init is None:
            # BB is of size 0 at axis 0
            self.bounding_box = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            self.points_list = list()

        # When there is two points to define the shape to aff (disconnect to any learning its only for UI)
        elif bounding_box_init.shape[0] == 2:
            x_min = bounding_box_init[0, 0]
            y_min = bounding_box_init[0, 1]
            x_max = bounding_box_init[1, 0]
            y_max = bounding_box_init[1, 1]
            # create the BB with the right size using 2 points
            self.bounding_box = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
            print(self.bounding_box)

            # get xmin/ymax and xmax/ymin points which represent the two additional point needed to define a rectangle
            # useful latter with event
            self.points_list = [self.bounding_box[1, :], self.bounding_box[3, :]]

            # plot xmin/ymin and xmax/ymax points
            self.ax.scatter(self.bounding_box[0, 0], self.bounding_box[0, 1])
            self.ax.scatter(self.bounding_box[2, 0], self.bounding_box[2, 1]

        # When there is four points to define the shape in the bounding_box_init param, only happen in update_bounding_box
        elif bounding_box_init.shape[0] == 4:
            # already have a BB with the right size
            self.bounding_box = bounding_box_init

            # get xmin/ymax and xmax/ymin points which represent the two additional point needed to define a rectangle
            # useful latter with event
            self.points_list = [self.bounding_box[1, :], self.bounding_box[3, :]]

            # plot xmin/ymin and xmax/ymax points
            self.ax.scatter(self.bounding_box[0, 0], self.bounding_box[0, 1])
            self.ax.scatter(self.bounding_box[2, 0], self.bounding_box[2, 1])
        else:
            print("Format error for bounding_box_init")

        # determine the width and height of the BB to draw on the image
        self.width = self.bounding_box[3, 0] - self.bounding_box[1, 0]
        self.height = self.bounding_box[2, 1] - self.bounding_box[0, 1]

        #self.bounding_box[0, :] get the coordinate of the starting points needed to draw our BB
        rect = patches.Rectangle(self.bounding_box[0, :], self.width, self.height,
                                 linewidth=1, edgecolor='b', facecolor='None')
        self.ax.add_patch(rect)
        fig.canvas.draw_idle()

        # add event to interact with the Image
        self.press = False
        self.move = False
        self.c1 = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press_button)
        self.c2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_press_r)
        self.c3 = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release_button)
        self.c4 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    """
        each time we click on a new point, we need to determine again the two points we will use to create our BB.
        We keep only the absolute xmin, ymin, xmax and ymax, which mean we can ONLY increase the size of our BB in the 
        four directions.
        
    """
    def update_bounding_box(self):
        if len(self.points_list) == 0:
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
        else:
            random_point = next(iter(self.points_list))
            x_min = random_point[0]
            y_min = random_point[1]
            x_max = random_point[0]
            y_max = random_point[1]

        for x, y in self.points_list:
            x_min = int(min(x_min, x))
            y_min = int(min(y_min, y))
            x_max = int(max(x_max, x))
            y_max = int(max(y_max, y))

        self.bounding_box = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
        self.width = x_max - x_min
        self.height = y_max - y_min

    # Update the drawing of the BB acordingly to the new xmin, ymin, xmax and ymax
    def update_plot(self, fig, new_point):
        self.ax.patches[0].set_xy(self.bounding_box[0, :])
        self.ax.patches[0].set_width(self.width)
        self.ax.patches[0].set_height(self.height)
        self.ax.patches[0].set_edgecolor("r")
        if new_point is None:
            self.ax.collections.pop()
        else:
            self.ax.scatter(new_point[0], new_point[1])
        fig.canvas.draw_idle()

    #get the xmin, ymin, xmax and ymax of the BB
    def get_bounding_box(self, format="ul_dl_dr_ur"):
        if format == "ul_dl_dr_ur":
            return self.bounding_box
        elif format == "ul_br":
            return np.array([self.bounding_box[0, :], self.bounding_box[2, :]])
        elif format == "bl_width_height":
            return [self.bounding_box[0, :], self.width, self.height]

    #No usage?
    def get_points_set(self):
        return self.points_list
    """
        key event capture and resulting interaction/processing
    """
    def on_click(self, event):
        x, y = (int(event.xdata), int(event.ydata))
        self.points_list.append((x, y))
        self.update_bounding_box()
        self.update_plot(fig=event.canvas.figure, new_point=(x, y))

    def on_press_r(self, event):
        if event.key == "r":
            self.points_list.pop()
            self.update_bounding_box()
            self.update_plot(fig=event.canvas.figure, new_point=None)

    def on_press_button(self, event):
        self.press = True

    def on_move(self, event):
        if self.press:
            self.move = True

    def on_release_button(self, event):
        if self.press and not self.move:
            self.on_click(event)
        self.press = False
        self.move = False
