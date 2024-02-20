"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import pickle as pkl

import torch
from torch.utils.data import Dataset
from PIL import Image

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOCAction(Dataset):
    """Pascal VOC2012 Action Dataset.

    Parameters
    ----------
    root : str, default '~/data/VOCdevkitc'
        Path to folder storing the dataset.
    split : str, default 'train'
        Candidates can be: 'train', 'val', 'trainval', 'test'.
    index_map : dict, default None
        In default, the 11 classes are mapped into indices from 0 to 10. We can
        customize it by providing a str to int dict specifying how to map class
        names to indicies. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extreamly large.
    """
    CLASSES = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
               'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'other')

    def __init__(self, root='',
                 split='train', index_map=None, preload_label=True,
                 augment_box=True, load_box=True, random_cls=False):
        super(VOCAction, self).__init__()
        self.num_class = len(self.CLASSES)
        self._im_shapes = {}
        self._root = root
        self._augment_box = augment_box
        self._load_box = load_box
        self._random_cls = random_cls
        self._split = split
        if self._split.lower() == 'val':
            self._jumping_start_pos = 307
        elif self._split.lower() == 'test':
            self._jumping_start_pos = 613
        else:
            self._jumping_start_pos = 0
        self._items = self._load_items(split)
        self._anno_path = os.path.join(self._root, 'Annotations', '{}.xml')
        self._box_path = os.path.join(self._root, 'Boxes', '{}.pkl')
        self._image_path = os.path.join(self._root, 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

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
        img = np.array(img)

        if self._random_cls:
            for i, cls in enumerate(label[:, 5:]):
                candidate_cls = np.array(np.where(cls == 1)).reshape((-1,))
                label[i, 4] = np.random.choice(candidate_cls)
        if self._augment_box:
            h, w, _ = img.shape
            if self._split == 'train':
                self.my_transform = A.Compose(
                    [A.Resize(224, 224),
                     A.RandomBrightnessContrast(p=0.2),
                     A.SafeRotate(15, p=0.2),
                     A.HorizontalFlip(p=0.2),
                     A.ColorJitter(brightness=0.3, p=0.2),
                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ToTensorV2()
                     ],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                )
            else:
                self.my_transform = A.Compose(
                    [A.Resize(224, 224),
                     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                     ToTensorV2()
                     ],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
                )
        if self._load_box:
            box_path = self._box_path.format(img_id)
            with open(box_path, 'rb') as f:
                box = pkl.load(f)
                box = torch.tensor(box, dtype=torch.float32)

            label = torch.tensor(label, dtype=torch.int32)
            h_box = label[0][:4]

            if self._augment_box:
                box_hbox_pose = torch.concatenate((box, h_box.reshape(1, -1)), dim=0)
                # box_hbox_pose = torch.concatenate((box_hbox, pose), dim=0)
                labels = np.ones_like(box_hbox_pose[:, 0])

                corrected_bboxes = [self.correct_bbox_order(self.correct_bbox_coordinates(bbox, img)) for bbox in box_hbox_pose]

                transformed = self.my_transform(image=img, bboxes=corrected_bboxes, labels=labels)

                img = transformed['image']
                box_t = transformed['bboxes'][:len(box)]
                hbox_t = transformed['bboxes'][len(box):len(box)+1]
                pose_t = transformed['bboxes'][len(box) + 1:]

                # self.visualize(img, box_t)
                # self.visualize(img, hbox_t)

                _, img_width, img_height = img.shape

                box_t = self.normalize_bbox(box_t, img_width, img_height)
                hbox_t = self.normalize_bbox(hbox_t, img_width, img_height)
                pose_t = self.normalize_bbox(pose_t, img_width, img_height)

                box = torch.tensor(box_t, dtype=torch.float32)
                h_box = torch.tensor(hbox_t, dtype=torch.float32)
                pose = torch.tensor(pose_t, dtype=torch.float32)

            return img, label, h_box, box, pose


            return img, label, box
        return img, label

    def _load_items(self, split):
        """Load individual image indices from split."""
        ids = []
        set_file = os.path.join(self._root, 'ImageSets', 'Action', split + '.txt')
        with open(set_file, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
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

            cls_id = -1
            act_cls = obj.find('actions')
            cls_array = [0] * len(self.classes)
            if idx < self._jumping_start_pos:
                # ignore jumping class according to voc offical code
                cls_array[0] = -1
            if act_cls is not None:
                for i, cls_name in enumerate(self.classes):
                    is_action = float(act_cls.find(cls_name).text)
                    if is_action > 0.5:
                        cls_id = i
                        cls_array[i] = 1
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

    def correct_bbox_order(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        if x_max < x_min:
            x_min, x_max = x_max, x_min  # Swap x_min and x_max
        elif x_max == x_min:
            x_min -= 1

        if y_max < y_min:
            y_min, y_max = y_max, y_min  # Swap y_min and y_max
        elif y_max == y_min:
            y_min -= 1

        return x_min, y_min, x_max, y_max

    def correct_bbox_coordinates(self, bbox, image):
        x_min, y_min, x_max, y_max = bbox
        image_y, image_x, _ = image.shape

        if x_max > image_x:
            x_max = image_x
        if y_max > image_y:
            y_max = image_y

        return x_min, y_min, x_max, y_max

    def visualize(self, image, bboxes):
        img = image.copy()
        for bbox in bboxes:
            img = self.visualize_bbox(img, bbox)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def visualize_bbox(self, img, bbox, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = np.array(bbox)

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)
        return img

    def normalize_bbox(self, bbox, width, height):
        """Normalize bounding box coordinates."""
        box_list= []
        for box in bbox:
            x1, y1, x2, y2 = box
            n_box = [x1 / width, y1 / height, x2 / width, y2 / height]
            box_list.append(n_box)
        return box_list