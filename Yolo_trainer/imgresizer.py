import os, sys, shutil, glob, cv2
import numpy as np
from PIL import Image
from torchvision import transforms

#img_size=416


input_path = 'data/images-kitti-jpg/'
output_path = 'data/artifacts/images/'


#img = cv2.imread('data/modified/train/images-kitti/000003.png', cv2.IMREAD_UNCHANGED)

def make_square(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128))])
    image = img_transforms(img)
    return image


#input = 'data/modified/000414.png'
#img = Image.open(input)
#img_size = max(img.size[0], img.size[1])
#img_size = 416
#resized = make_square(img)
#resized.show()

for f in os.listdir(input_path):
    test_image = Image.open(input_path + f)
    img_size = max(test_image.size[0],test_image.size[1])
    resized = make_square(test_image)
    resized.save(output_path + f)


#for f in os.listdir(input_path):
    #img = cv2.imread(input_path + f, cv2.IMREAD_UNCHANGED)
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #cv2.imwrite(output_path + f, resized)
