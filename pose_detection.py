import os

import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import pickle as pkl
from PIL import Image

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def crop_image(image_path, coordinates):

    img = Image.open(image_path)
    x, y, x2, y2 = coordinates

    cropped_image = img.crop((x, y, x2, y2))
    return cropped_image

root = 'Datasets/Stanford40/XMLAnnotations'
root_pose = 'Datasets/Stanford40/PoseAnno'
list_items = os.listdir(root)

num_workers = 0
batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a model
model = YOLO('weights/pose/yolov8l-pose.pt')
model.to(device)
model.fuse()

for item in list_items:
    anno_path = os.path.join(root, '{}')
    image_path = 'Datasets/Stanford40/JPEGImages/{}.jpg'

    each_item = anno_path.format(item)

    tree = ET.parse(each_item)
    xml_box = tree.find('object').find('bndbox')
    xmin = (float(xml_box.find('xmin').text) - 1)
    ymin = (float(xml_box.find('ymin').text) - 1)
    xmax = (float(xml_box.find('xmax').text) - 1)
    ymax = (float(xml_box.find('ymax').text) - 1)

    anno = [xmin, ymin, xmax, ymax]
    name, _ = item.split('.')
    image = image_path.format(name)

    croped_image = crop_image(image, anno)

    outt = model(croped_image)
    keypoints = outt[0].keypoints.data[0].detach().cpu().numpy()

    keypoints[:, 0] += xmin
    keypoints[:, 1] += ymin
    #
    keypoint_names = [
        'Nose', 'REye', 'LEye', 'REer', 'LEar', 'RShoulder', 'LShoulder',
        'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RHip', 'LHip',
        'RKnee', 'LKnee', 'RAnkle', 'LAnkle'
    ]
    my_dict = {}

    for key_name, (xx, yy, conf) in zip(keypoint_names, keypoints):
        # ax.annotate(key_name, (xx, yy), textcoords="offset points", xytext=(0, 10), ha='center', c='darkblue')
        my_dict[key_name] = (xx, yy, conf)

    name_to_save = '{}.pkl'.format(name)
    pickle_file_name = os.path.join(root_pose, name_to_save)
    with open(pickle_file_name, 'wb') as pickle_file:
        pkl.dump(my_dict, pickle_file)


