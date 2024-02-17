import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pkl

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def select_random_box(bbox):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the width and height of each of the 9 pieces
    sub_width = width / 3
    sub_height = height / 3

    # Generate the coordinates for the 9 pieces
    pieces = []
    for i in range(3):
        for j in range(3):
            x_start = x_min + i * sub_width
            y_start = y_min + j * sub_height
            x_end = x_start + sub_width
            y_end = y_start + sub_height
            pieces.append([x_start, y_start, x_end, y_end])

    return random.choice(pieces)

def calculate_pose_regions(keypoints, image, human_box):

    x_img, y_img = image.shape[1], image.shape[0]

    pose_regions = {
        #'Body': [keypoints['LShoulder'], keypoints['RShoulder'], keypoints['LHip'], keypoints['RHip']],
        'Left hand': [keypoints['LElbow'], keypoints['LWrist']],
        'Right hand': [keypoints['RElbow'], keypoints['RWrist']],
        'Left elbow': [keypoints['LWrist'], keypoints['LElbow'], keypoints['LShoulder']],
        'Right elbow': [keypoints['RWrist'], keypoints['RElbow'], keypoints['RShoulder']],
        'Left foot': [keypoints['LKnee'], keypoints['LAnkle']],
        'Right foot': [keypoints['RKnee'], keypoints['RAnkle']],
        'Left knee': [keypoints['LAnkle'], keypoints['LKnee'], keypoints['LHip']],
        'Right knee': [keypoints['RAnkle'], keypoints['RKnee'], keypoints['RHip']]
    }

    pose_boxes = {}
    for region, key_points in pose_regions.items():
        if len(key_points) == 2:

            if key_points[1][2] < 0.5:
                min_x, min_y, max_x, max_y = select_random_box(human_box)

            # When i have 1 keypoint (Hand, Ankle)
            elif key_points[0][2] < 0.5:
                xmin_human, ymin_human, xmax_human, ymax_human = human_box

                # Calculate width and height of the human bounding box
                width_human = xmax_human - xmin_human
                height_human = ymax_human - ymin_human

                # Calculate aspect ratio of human bounding box
                aspect_ratio_human = width_human / height_human

                # Hand keypoint coordinates
                hand_x, hand_y = key_points[1][:2]

                width_hand = 0.2 * width_human  # Adjust 0.2 based on desired hand size relative to human
                height_hand = width_hand / aspect_ratio_human

                min_x = int(hand_x - width_hand / 2)
                min_y = int(hand_y - height_hand / 2)
                max_x = int(hand_x + width_hand / 2)
                max_y = int(hand_y + height_hand / 2)

            else:
                # Assuming key_points[1] is the wrist and key_points[0] is the elbow
                wrist_x, wrist_y = key_points[1][:2]
                elbow_x, elbow_y = key_points[0][:2]

                # Calculate the distance between wrist and elbow
                distance = np.linalg.norm((wrist_x - elbow_x, wrist_y - elbow_y))

                # Define a factor to determine the box size based on the wrist-elbow distance
                if region in ['Left hand', 'Right hand']:
                    box_size_factor = 1.2
                elif region in ['Left foot', 'Right foot']:
                    box_size_factor = 0.85
                else:
                    box_size_factor = 1.5

                # Calculate the half-size of the bounding box based on the wrist size
                half_size = distance * box_size_factor / 2

                # Calculate bounding box coordinates
                min_x = int(wrist_x - half_size)
                min_y = int(wrist_y - half_size)
                max_x = int(wrist_x + half_size)
                max_y = int(wrist_y + half_size)

            if min_x < 0:
                min_x = 0

            if min_y < 0:
                min_y = 0

            if max_x > x_img:
                max_x = x_img

            if max_y > y_img:
                max_y = y_img

            if min_x == max_x:
                max_x = max_x + 10
                max_y = max_y + 10

            pose_boxes[region] = [(min_x, min_y), (max_x, max_y)]

        if len(key_points) == 3:
            if key_points[1][2] < 0.5:
                min_x, min_y, max_x, max_y = select_random_box(human_box)

            elif key_points[0][2] < 0.5 and key_points[2][2] < 0.5:
                xmin_human, ymin_human, xmax_human, ymax_human = human_box

                # Calculate width and height of the human bounding box
                width_human = xmax_human - xmin_human
                height_human = ymax_human - ymin_human

                # Calculate aspect ratio of human bounding box
                aspect_ratio_human = width_human / height_human

                # Hand keypoint coordinates
                elbow_x, elbow_y = key_points[1][:2]

                width_hand = 0.2 * width_human  # Adjust 0.2 based on desired hand size relative to human
                height_hand = width_hand / aspect_ratio_human

                min_x = int(elbow_x - width_hand / 2)
                min_y = int(elbow_y - height_hand / 2)
                max_x = int(elbow_x + width_hand / 2)
                max_y = int(elbow_y + height_hand / 2)

            elif key_points[2][2] < 0.5:
                elbow_x, elbow_y = key_points[1][:2]
                wrist_x, wrist_y = key_points[0][:2]

                distance = np.linalg.norm((elbow_x - wrist_x, elbow_y - wrist_y))

                box_size_factor = 1.0

                # Calculate the half-size of the bounding box based on the wrist size
                half_size = distance * box_size_factor / 2

                # Calculate bounding box coordinates
                min_x = int(elbow_x - half_size)
                min_y = int(elbow_y - half_size)
                max_x = int(elbow_x + half_size)
                max_y = int(elbow_y + half_size)

            elif key_points[0][2] < 0.5:
                elbow_x, elbow_y = key_points[1][:2]
                shoulder_x, shoulder_y = key_points[2][:2]

                distance = np.linalg.norm((elbow_x - shoulder_x, elbow_y - shoulder_y))

                box_size_factor = 1.0

                # Calculate the half-size of the bounding box based on the wrist size
                half_size = distance * box_size_factor / 2

                # Calculate bounding box coordinates
                min_x = int(elbow_x - half_size)
                min_y = int(elbow_y - half_size)
                max_x = int(elbow_x + half_size)
                max_y = int(elbow_y + half_size)
            else:

                hand_x, hand_y = key_points[0][:2]
                elbow_x, elbow_y = key_points[1][:2]
                shoulder_x, shoulder_y = key_points[2][:2]

                # Calculate distances between elbow and hand, and between elbow and shoulder
                distance_hand_elbow = np.linalg.norm((hand_x - elbow_x, hand_y - elbow_y))
                distance_elbow_shoulder = np.linalg.norm((shoulder_x - elbow_x, shoulder_y - elbow_y))

                box_size_factor_hand = 0.8  # Adjust this factor as needed for the hand distance
                box_size_factor_shoulder = 0.8

                if region in ['Left knee', 'Right knee']:
                    box_size_factor_hand = 0.7
                    box_size_factor_shoulder = 0.7


                # Calculate the half-size of the bounding box based on the distances
                half_size_hand = distance_hand_elbow * box_size_factor_hand / 2
                half_size_shoulder = distance_elbow_shoulder * box_size_factor_shoulder / 2

                # Use the larger size to ensure that the box covers both the hand and shoulder regions
                box_size = max(half_size_hand, half_size_shoulder)

                # Calculate bounding box coordinates around the elbow
                min_x = int(elbow_x - box_size)
                min_y = int(elbow_y - box_size)
                max_x = int(elbow_x + box_size)
                max_y = int(elbow_y + box_size)

            if min_x < 0:
                min_x = 0

            if min_y < 0:
                min_y = 0

            if max_x > x_img:
                max_x = x_img

            if max_y > y_img:
                max_y = y_img

            pose_boxes[region] = [(min_x, min_y), (max_x, max_y)]

        # Not Important Body
        # elif len(key_points) == 4:
        #     min_x = min(key_points[0][0], key_points[2][0])
        #     min_y = min(key_points[0][1], key_points[2][1])
        #     max_x = max(key_points[1][0], key_points[3][0])
        #     max_y = max(key_points[1][1], key_points[3][1])
        #     pose_boxes[region] = [(min_x, min_y), (max_x, max_y)]

    return pose_boxes

def visualize_items(image ,pose_boxes, h_bbox):
    fig, ax = plt.subplots()

    ax.imshow(image)
    for pose in pose_boxes.values():

        x1, y1, x2, y2 = pose[0][0], pose[0][1], pose[1][0], pose[1][1]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def plot_keypoints_with_boxes(keypoints, boxes, image):
    x = [point[0] for point in keypoints.values()]
    y = [point[1] for point in keypoints.values()]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='r', marker='o')

    for box_type, box_coordinates in boxes.items():
        rect = patches.Rectangle(
            box_coordinates[0],
            box_coordinates[1][0] - box_coordinates[0][0],
            box_coordinates[1][1] - box_coordinates[0][1],
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.annotate(box_type, box_coordinates[0], textcoords="offset points", xytext=(0, -10), ha='center', color='b')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.imshow(image)
    plt.show()

# path = 'Datasets/Stanford40/JPEGImages/*.jpg'
# all_images = np.array(list(glob.iglob(path)))

# image_idx = np.random.choice(len(all_images), 1)
# image_chosen_path = all_images[image_idx.item()]

pose_path = 'Datasets/Stanford40/PoseAnno/*.pkl'
all_poses = np.array(list(glob.iglob(pose_path)))

root_pose = 'Datasets/Stanford40/PoseBoxes'
root_image = 'Datasets/Stanford40/JPEGImages'
root_xml = 'Datasets/Stanford40/XMLAnnotations'

# pose_boxes = calculate_pose_regions(keypoints_dict, image, h_bbox)
# plot_keypoints_with_boxes(keypoints_dict, pose_boxes, image)

for item in all_poses:

    # Vis
    item = np.random.choice(all_poses)

    print(item)
    with open(item, 'rb') as pickle_file:
        keypoints_dict = pkl.load(pickle_file)

    name, _ = item.split('.')
    name = name.split('\\')[1]
    name_to_save = '{}.pkl'.format(name)
    image_name = '{}.jpg'.format(name)
    image_name_xml = '{}.xml'.format(name)

    image_file_name = os.path.join(root_image, image_name)
    image = plt.imread(image_file_name)

    each_item = os.path.join(root_xml, image_name_xml)

    tree = ET.parse(each_item)
    xml_box = tree.find('object').find('bndbox')
    xmin_b = (float(xml_box.find('xmin').text) - 1)
    ymin_b = (float(xml_box.find('ymin').text) - 1)
    xmax_b = (float(xml_box.find('xmax').text) - 1)
    ymax_b = (float(xml_box.find('ymax').text) - 1)

    h_bbox = (xmin_b, ymin_b, xmax_b, ymax_b)

    pose_boxes = calculate_pose_regions(keypoints_dict, image, h_bbox)

    # Vis
    visualize_items(image, pose_boxes, h_bbox)
    break
    # pickle_file_name = os.path.join(root_pose, name_to_save)
    # with open(pickle_file_name, 'wb') as pickle_file:
    #     pkl.dump(list(pose_boxes.values()), pickle_file)


