# Imports
from models import *
from utils import *

import os, sys, time, datetime, random
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.datasets.folder import default_loader

import matplotlib.pyplot as plt
import imageio
import matplotlib.patches as patches
from PIL import Image


# Names of the classes
COCO_INSTANCE_CATEGORY_NAMES = [
'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# Load and evaluate the network
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval() # Set network in eval mode (e.g. disable dropout, batchnorm, ...)
Tensor = torch.cuda.FloatTensor


tfs = transforms.Compose([transforms.ToTensor()])

def detect_image(img):
    image_tensor = tfs(img)
    img_th = Variable(image_tensor.type(Tensor))
    
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model([img_th])[0] # Output is a List[Dict], one element for each input image
    return detections


import cv2
from IPython.display import clear_output

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
from sort import *
#vid = cv2.VideoCapture(videopath)
vid = cv2.VideoCapture(0)
mot_tracker = Sort() 


# Perform detection operation on each frame, and draw bounding box with labels
img_array = []
# Realize a 30sec c.a. video
for ii in range(500):
    ret, frame = vid.read()
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    img = np.array(pilimg)
    
    if detections is not None:
        # FasterRCNN gives a list as output, but we need a tensor to give it to SORT 
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        high_prob_obj = []
        # Filter out all objects having low probability
        for j, scr in enumerate(scores):
            if scores[j] > 0.7:
                kk = torch.tensor([boxes[j,0], boxes[j,1], boxes[j,2], boxes[j,3], scores[j], scores[j], labels[j]])
                high_prob_obj.append(kk)
        # Create a tensor from the list, "unrolling" along dim = 0
        if len(high_prob_obj)>0:
            obj_tensor = torch.stack(high_prob_obj, dim = 0)
        
            # Pass the tensor with detections to the SORT
            tracked_objects = mot_tracker.update(obj_tensor.cpu())


            unique_labels = obj_tensor[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            # Display the bounding boxes, showing the class and the ID of the detected object
            # Pay attention: the object shown are a little bit less than the detected ones. If you use this network for e.g. autonomous
            #                driving, give obj_tensor to the controller (contains ALL the detected objects)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = COCO_INSTANCE_CATEGORY_NAMES[int(cls_pred.item())]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - 35), (int(x1) + len(cls)*19+60, int(y1)), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        else:
            print('No object detected')


    height, width, layers = img.shape
    size = (width,height)
    # update the list containing the frames with detection
    img_array.append(frame)  
    print('Frame:', ii)
    print()


# Save the video as a frame sequence
out = cv2.VideoWriter('RCNN_tracking_save_from_live.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size) # 3rd entry is the number of fps
 
length = len(img_array)
for i in range(length):
    out.write(img_array[i])
out.release()

# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()