import os, sys, shutil, glob, cv2
import numpy as np
from PIL import Image

img_size=416
dim = (416, 128)

input_path = 'data/modified/images-kitti_orig/'
output_path = 'data/modified/images-kitti-resized/'


def make_square(im, min_size=416, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    #size = max(min_size, x, y)
    size = 416
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


for f in os.listdir(input_path):
    test_image = Image.open(input_path + f)
    resized = make_square(test_image)
    #img = cv2.imread(input_path + f, cv2.IMREAD_UNCHANGED)
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #cv2.imwrite(output_path + f, resized)
    resized.save(output_path + f)
