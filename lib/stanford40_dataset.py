"""Stanford 40 Actions dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torch

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import pickle as pkl

## Albumenation
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

class Stanford40Action(Dataset):

    CLASSES = ("applauding", "blowing_bubbles", "brushing_teeth", "cleaning_the_floor", "climbing", "cooking",
               "cutting_trees", "cutting_vegetables", "drinking", "feeding_a_horse", "fishing", "fixing_a_bike",
               "fixing_a_car", "gardening", "holding_an_umbrella", "jumping", "looking_through_a_microscope",
               "looking_through_a_telescope", "playing_guitar", "playing_violin", "pouring_liquid", "pushing_a_cart",
               "reading", "phoning", "riding_a_bike", "riding_a_horse", "rowing_a_boat", "running",
               "shooting_an_arrow", "smoking", "taking_photos", "texting_message", "throwing_frisby",
               "using_a_computer", "walking_the_dog", "washing_dishes", "watching_TV", "waving_hands",
               "writing_on_a_board", "writing_on_a_book")

    def __init__(self, root='', transform = None,
                 split='train', index_map=None, preload_label=True,
                 augment_box=False, load_box=False):
        super(Stanford40Action, self).__init__()
        self.num_class = len(self.classes)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._load_box = load_box
        self._augment_box = augment_box
        self._split = split
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'XMLAnnotations', '{}.xml')
        self._image_path = os.path.join(self._root, 'JPEGImages', '{}.jpg')
        self._box_path = os.path.join(self._root, 'Boxes', '{}.pkl')
        self._pose_path = os.path.join(self._root, 'PoseBoxes', '{}.pkl')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None
        self.transform = transform

        self.my_transform = None

    def __str__(self):
        return self.__class__.__name__ + '(' + self._split + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def img_path(self, idx):
        img_id = self._items[idx]
        return self._image_path.format(img_id)

    def save_boxes(self, idx, boxes):
        img_id = self._items[idx]
        box_path = self._box_path.format(img_id)
        with open(box_path, 'wb') as f:
            pkl.dump(boxes, f)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = Image.open(img_path).convert('RGB')

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.transform is not None:
            if self._split == 'train':
                self.my_transform = A.Compose(
                    [A.Resize(300, 300),
                     A.RandomBrightnessContrast(p=0.3),
                     A.SafeRotate(15, p=0.3),
                     A.HorizontalFlip(p=0.3),
                     A.ColorJitter(brightness=0.3, p=0.3),
                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ToTensorV2()
                     ],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                )
            else:
                self.my_transform = A.Compose(
                    [A.Resize(300, 300),
                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ToTensorV2()
                     ],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                )

        if self._load_box:
            box_path = self._box_path.format(img_id)
            pose_path = self._pose_path.format(img_id)
            with open(box_path, 'rb') as f:
                box = pkl.load(f)
                box = torch.tensor(box, dtype=torch.float32)

            with open(pose_path, 'rb') as m:
                pose = pkl.load(m)
                pose = [[item for sublist in sublist_tuple for item in sublist] for sublist_tuple in pose]
                pose = torch.tensor(pose, dtype=torch.float32)

            label = torch.tensor(label, dtype=torch.int32)
            h_box = label[0][:4]

            if self._augment_box:
                img = np.array(img)

                box_hbox = torch.concatenate((box, h_box.reshape(1, -1)), dim=0)
                box_hbox_pose = torch.concatenate((box_hbox, pose), dim=0)
                labels = np.ones_like(box_hbox_pose[:, 0])

                corrected_bboxes = [self.correct_bbox_order(self.correct_bbox_coordinates(bbox, img)) for bbox in box_hbox_pose]

                transformed = self.my_transform(image=img, bboxes=corrected_bboxes, labels=labels)

                img = transformed['image']
                box_t = transformed['bboxes'][:len(box)]
                hbox_t = transformed['bboxes'][len(box):len(box)+1]
                pose_t = transformed['bboxes'][len(box) + 1:]

                # self.visualize(img, box_t)
                # self.visualize(img, hbox_t)

                _, img_width, img_hight = img.shape

                box_t = self.normalize_bbox(box_t, img_width, img_hight)
                hbox_t = self.normalize_bbox(hbox_t, img_width, img_hight)
                pose_t = self.normalize_bbox(pose_t, img_width, img_hight)

                box = torch.tensor(box_t, dtype=torch.float32)
                h_box = torch.tensor(hbox_t, dtype=torch.float32)
                pose = torch.tensor(pose_t, dtype=torch.float32)

            return img, label, h_box, box, pose
        return img, label

    def _load_items(self, split):
        """Load individual image indices from split."""
        ids = []
        set_file = os.path.join(self._root, 'ImageSplits', split + '.txt')
        with open(set_file, 'r') as f:
            # remove file extensions
            ids += [line.strip().rsplit(sep='.', maxsplit=1)[0] for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name != 'person':
                continue

            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))

            act_cls_name = obj.find('action').text
            cls_id = self.index_map[act_cls_name]
            cls_array = [0] * len(self.classes)
            cls_array[cls_id] = 1
            anno = [xmin, ymin, xmax, ymax, cls_id]
            anno.extend(cls_array)
            label.append(anno)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


    ## Albumenation
    def visualize_bbox(self, img, bbox, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = np.array(bbox)

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=thickness)
        return img

    def visualize(self, image, bboxes):
        img = image.copy()
        for bbox in bboxes:
            img = self.visualize_bbox(img, bbox)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def correct_bbox_order(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min:
            x_min, x_max = x_max, x_min  # Swap x_min and x_max

        if y_max <= y_min:
            y_min, y_max = y_max, y_min  # Swap y_min and y_max

        return x_min, y_min, x_max, y_max

    def correct_bbox_coordinates(self, bbox, image):

        x_min, y_min, x_max, y_max = bbox
        image_y, image_x, _ = image.shape

        if x_max > image_x:
            x_max = image_x

        if y_max > image_y:
            y_max = image_y

        return x_min, y_min, x_max, y_max

    def normalize_bbox(self, bbox, width, height):
        """Normalize bounding box coordinates."""
        box_list= []
        for box in bbox:
            x1, y1, x2, y2 = box
            n_box = [x1 / width, y1 / height, x2 / width, y2 / height]
            box_list.append(n_box)
        return box_list