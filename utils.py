import matplotlib.patches as patches
from tensorflow import keras
import numpy as np
import tensorflow as tf


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(units=84, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(units=4, activation=tf.sigmoid)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mse',
                  metrics=['mse'])

    return model

class BoundingBoxDraw_app():

    def __init__(self, ax, fig, bounding_box_init=None):

        self.ax = ax

        if bounding_box_init is None:
            self.bounding_box = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            self.points_list = list()
        elif bounding_box_init.shape[0] == 2:
            x_min = bounding_box_init[0, 0]
            y_min = bounding_box_init[0, 1]
            x_max = bounding_box_init[1, 0]
            y_max = bounding_box_init[1, 1]
            self.bounding_box = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
            print(self.bounding_box)
            self.points_list = [self.bounding_box[1, :], self.bounding_box[3, :]]
            self.ax.scatter(self.bounding_box[0, 0], self.bounding_box[0, 1])
            self.ax.scatter(self.bounding_box[2, 0], self.bounding_box[2, 1])
        elif bounding_box_init.shape[0] == 4:
            self.bounding_box = bounding_box_init
            self.points_list = [self.bounding_box[1, :], self.bounding_box[3, :]]
            self.ax.scatter(self.bounding_box[0, 0], self.bounding_box[0, 1])
            self.ax.scatter(self.bounding_box[2, 0], self.bounding_box[2, 1])
        else:
            print("Format error for bounding_box_init")

        self.width = self.bounding_box[3, 0] - self.bounding_box[1, 0]
        self.height = self.bounding_box[2, 1] - self.bounding_box[0, 1]

        rect = patches.Rectangle(self.bounding_box[0, :], self.width, self.height,
                                 linewidth=1, edgecolor='b', facecolor='None')
        self.ax.add_patch(rect)
        fig.canvas.draw_idle()

        self.press = False
        self.move = False
        self.c1 = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press_button)
        self.c2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_press_r)
        self.c3 = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release_button)
        self.c4 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

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

    def get_bounding_box(self, format="ul_dl_dr_ur"):
        if format == "ul_dl_dr_ur":
            return self.bounding_box
        elif format == "ul_br":
            return np.array([self.bounding_box[0, :], self.bounding_box[2, :]])
        elif format == "bl_width_height":
            return [self.bounding_box[0, :], self.width, self.height]

    def get_points_set(self):
        return self.points_list

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