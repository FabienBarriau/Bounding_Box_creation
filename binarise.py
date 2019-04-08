from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import utils as gui
import os

INPUT_FILE = "Image/"
OUTPUT_FILE = "Binary/"

names_list = list()
for name in os.listdir(INPUT_FILE):
    extension = name.split(".")[1]
    if extension.lower() in ["jpg", "png"]:
        names_list.append(name)

bb_list = list()

for c, name in enumerate(names_list):
    im = Image.open(INPUT_FILE + name).convert("RGB")
    # Create the figure
    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    ax.imshow(im)
    plt.get_current_fig_manager().window.state('zoomed')
    bb_app = gui.BoundingBoxDraw_app(ax, fig)
    plt.show(fig)
    bb = bb_app.get_bounding_box(format="ul_br")
    #Create binary image
    bin = Image.new('RGB', im.size, (0, 0, 0))
    draw = ImageDraw.Draw(bin)
    draw.rectangle([bb[0][0], bb[0][1], bb[1][0], bb[1][1]], fill=(255, 255, 255))
    fig_bin, ax_bin = plt.subplots(1, 1)
    ax_bin.imshow(bin)
    plt.show(fig_bin)
    bin.save(OUTPUT_FILE + name + ".jpg", "JPEG")





