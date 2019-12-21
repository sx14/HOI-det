import pickle
import numpy as np
from matplotlib import pyplot as plt
import cv2

body_parts = ["head",
              "left_hand",
              "right_hand",
              "hip",
              "left_leg",
              "right_leg"]


key_points = ["nose",
              "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

all_part_kps = {
    'left_leg': ['left_ankle'],
    'right_leg': ['right_ankle'],
    'left_hand': ['left_hand', 'left_wrist', 'left_elbow'],
    'right_hand': ['right_hand', 'right_wrist', 'right_elbow'],
    'hip': ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
    'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
}


def ext_skeleton_feature(key_points, obox, hbox):
    if key_points is None or key_points.sum() == 0:
        return [0] * (17 * 2 * 5)

    human_w = hbox[2] - hbox[0] + 1
    human_h = hbox[3] - hbox[1] + 1
    human_norm = (human_w + human_h) / 2.0
    ox1, oy1, ox2, oy2 = obox
    lf_top = [ox1, oy1]
    lf_bot = [ox1, oy2]
    rt_top = [ox2, oy1]
    rt_bot = [ox2, oy2]
    center = [(ox1+ox2)/2.0,
              (oy1+oy2)/2.0]
    box_pts = np.array([lf_top, lf_bot, rt_top, rt_bot, center])
    key_point_xys = key_points[:, :2]
    feat = []
    for i in range(box_pts.shape[0]):
        box_pt = box_pts[i:i+1]
        curr_feat = (key_point_xys - box_pt) / human_norm
        feat += curr_feat.reshape(-1).tolist()
    return feat