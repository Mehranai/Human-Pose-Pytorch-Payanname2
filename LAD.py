import os
import glob
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_limb_direction_vectors(keypoint_positions):
    limb_keypoints = [
        (0, 1),  # Torso
        (1, 2),  # Head
        (1, 3),  # Shoulder
        (3, 4),  # Left arm
        (3, 5),  # Right arm
        (4, 6),  # Left forearm
        (5, 7),  # Right forearm
        (1, 8),  # Left thigh
        (1, 11),  # Right thigh
        (8, 9),  # Left leg
        (11, 4)  # Right leg
        ]

    limb_directions = []

    for limb_pair in limb_keypoints:
        joint1, joint2 = limb_pair
        vec = keypoint_positions[joint2] - keypoint_positions[joint1]
        limb_directions.append(vec)

    return np.array(limb_directions)

def calculate_LAD(limb_directions):
    L = limb_directions.shape[0]
    LAD = np.zeros(int(L * (L - 1) / 2))
    index = 0

    for i in range(L):
        for j in range(i + 1, L):
            angle = np.dot(limb_directions[i], limb_directions[j]) / (
            np.linalg.norm(limb_directions[i]) * np.linalg.norm(limb_directions[j]))
            LAD[index] = angle
            index += 1
    return LAD

def calculate_iLAD(keypoint_positions):
    limb_keypoints = [(i, j) for i in range(13) for j in range(i + 1, 12)]
    limb_directions = []

    for limb_pair in limb_keypoints:
        joint1, joint2 = limb_pair
        vec = keypoint_positions[joint2] - keypoint_positions[joint1]
        limb_directions.append(vec)

    iLAD = np.zeros(int(len(limb_directions) * (len(limb_directions) - 1) / 2))
    index = 0

    for i in range(len(limb_directions)):
        for j in range(i + 1, len(limb_directions)):
            angle = np.dot(limb_directions[i], limb_directions[j]) / (np.linalg.norm(limb_directions[i]) * np.linalg.norm(limb_directions[j]))
            iLAD[index] = angle
            index += 1

    return iLAD

def visualize_lad(keypoints_d):

    nose = keypoints_d['Nose']
    r_shoulder = keypoints_d['RShoulder']
    l_shoulder = keypoints_d['LShoulder']
    mid_shulder = np.mean([r_shoulder, l_shoulder], axis=0)
    r_elbow = keypoints_d['RElbow']
    l_elbow = keypoints_d['LElbow']
    r_wrist = keypoints_d['RWrist']
    l_wrist = keypoints_d['LWrist']
    r_hip = keypoints_d['RHip']
    l_hip = keypoints_d['LHip']
    r_knee = keypoints_d['RKnee']
    l_knee = keypoints_d['LKnee']
    r_ankle = keypoints_d['RAnkle']
    l_ankle = keypoints_d['LAnkle']

    fig, ax = plt.subplots()

    # Bini
    arrow = patches.FancyArrowPatch(nose, mid_shulder, arrowstyle='->',
                                    mutation_scale=15, color='Blue', linewidth=2)
    ax.add_patch(arrow)

    # Shane
    arrow = patches.FancyArrowPatch(r_shoulder, l_shoulder, arrowstyle='->',
                                    mutation_scale=15, color='Orange', linewidth=2)
    ax.add_patch(arrow)

    # dast_rast
    arrow = patches.FancyArrowPatch(r_shoulder, r_elbow, arrowstyle='->',
                                    mutation_scale=15, color='Green', linewidth=2)
    ax.add_patch(arrow)

    # saed_rast
    arrow = patches.FancyArrowPatch(r_elbow, r_wrist, arrowstyle='->',
                                    mutation_scale=15, color='Red', linewidth=2)
    ax.add_patch(arrow)

    # dast_chap
    arrow = patches.FancyArrowPatch(l_shoulder, l_elbow, arrowstyle='->',
                                    mutation_scale=15, color='Purple', linewidth=2)
    ax.add_patch(arrow)

    # saed_chap
    arrow = patches.FancyArrowPatch(l_elbow, l_wrist, arrowstyle='->',
                                    mutation_scale=15, color='Brown', linewidth=2)
    ax.add_patch(arrow)

    # lagan_right
    arrow = patches.FancyArrowPatch(r_shoulder, r_hip, arrowstyle='->',
                                    mutation_scale=15, color='Pink', linewidth=2)
    ax.add_patch(arrow)

    # lagan_chap
    arrow = patches.FancyArrowPatch(l_shoulder, l_hip, arrowstyle='->',
                                    mutation_scale=15, color='Olive', linewidth=2)
    ax.add_patch(arrow)

    # paye_rast
    arrow = patches.FancyArrowPatch(r_hip, r_knee, arrowstyle='->',
                                    mutation_scale=15, color='Olive', linewidth=2)
    ax.add_patch(arrow)

    # paye_rast
    arrow = patches.FancyArrowPatch(r_knee, r_ankle, arrowstyle='->',
                                    mutation_scale=15, color='darkgreen', linewidth=2)
    ax.add_patch(arrow)

    # paye_chap
    arrow = patches.FancyArrowPatch(l_hip, l_knee, arrowstyle='->',
                                    mutation_scale=15, color='blueviolet', linewidth=2)
    ax.add_patch(arrow)

    # paye_chap
    arrow = patches.FancyArrowPatch(l_knee, l_ankle, arrowstyle='->',
                                    mutation_scale=15, color='peru', linewidth=2)
    ax.add_patch(arrow)



# Example keypoint positions (replace this with your actual keypoint positions)
keypoint_positions = np.array([
    [0, 2],  # Head
    [-1, 1],  # Shoulder
    [2, -2],  # Right sholder
    [-2, 1],  # Left arm
    [1, 1],  # Right arm
    [-2, 0],  # Left forearm
    [1, 0],  # Right forearm
    [-1, -2],  # Left thigh
    [0, -2],  # Right thigh
    [1, -2],  # Left leg
    [3, -2],   # Right leg
    [3, -2],    # Left ancle
    [3, -2],    # Right anckle
    ])

# Calculate LAD
limb_directions = calculate_limb_direction_vectors(keypoint_positions)
LAD = calculate_LAD(limb_directions)
print("Limb Angle Descriptor (LAD):", LAD)
print("LAD dimension:", len(LAD))

# Calculate iLAD
iLAD = calculate_iLAD(keypoint_positions)
print("Improved Limb Angle Descriptor (iLAD):", iLAD)
print("iLAD dimension:", len(iLAD))

# Visualize
pose_path = 'Datasets/Stanford40/PoseAnno/*.pkl'
all_poses = np.array(list(glob.iglob(pose_path)))

root_image = 'Datasets/Stanford40/JPEGImages'
all_items = os.listdir(root_image)
image_name = np.random.choice(all_items)
image_file_name = os.path.join(root_image, image_name)

item = np.random.choice(all_poses)

with open(item, 'rb') as pickle_file:
    keypoints_dict = pkl.load(pickle_file)

name, _ = item.split('.')
name = name.split('\\')[1]
image_name = '{}.jpg'.format(name)

image_file_name = os.path.join(root_image, image_name)
image = plt.imread(image_file_name)


# visualize_lad(keypoints_dict)

plt.imshow(image)
plt.show()
