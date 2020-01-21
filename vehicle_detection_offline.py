import torchvision
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import cv2
import matplotlib.pyplot as plt
import imageio
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

author = plt.figure() # Necessary for the creation of the video only

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
model.eval() # Set network in eval mode (e.g. disable dropout, batchnorm, ...)


def detection(img):
    # Draw detections on image
    tfs = transforms.Compose([transforms.ToTensor()])
    img_th = tfs(img)
    output = model([img_th])[0] # Output is a List[Dict], one element for each input image

    # We take each frame as an image; output var. is a dictionary containing info on where the box is,
    # which is the label of the detection and the score. We simply generate a new image placing informations
    # on top of the original image
    fig = plt.imshow(img)
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']
    for bbox, cls, prob in zip(boxes, labels, scores):
        if prob > 0.8:
            fig = plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=2, alpha=0.5)
                )
            fig = plt.gca().text(bbox[0], bbox[1] - 2,
                    '%s' % (COCO_INSTANCE_CATEGORY_NAMES[cls.item()]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
    #return the modified frame to be used for video creation
    return fig


# Load and normalize the image
reader = imageio.get_reader('C:/Users/giaco/Desktop/car_video_extremetrim.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
print(fps)
print()
framelist = []
#writer = imageio.get_writer('C:/Users/giaco/Desktop/output.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detection(frame) # We call our detect function (defined above) to detect the object on the frame.
    #writer.append_data(frame) # We add the next frame in the output video.
    framelist.append([frame])
    print(i) # We print the number of the processed frame.
#writer.close() # We close the process that handles the creation of the output video.

mywriter = animation.FFMpegWriter(fps=fps)
ani = animation.ArtistAnimation(author, framelist, interval=(1000//fps), blit=True)
ani.save('C:/Users/giaco/Desktop/dynamic_images.mp4', writer=mywriter)

plt.show()