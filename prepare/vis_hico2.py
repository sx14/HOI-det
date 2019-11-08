import pickle
import numpy as np
from utils.show_box import show_boxes


def get_box(pts):
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    vsum = 0.0
    for i in range(len(pts)):
        x, y, v = pts[i, :]
        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x, xmax)
        ymax = max(y, ymax)
        vsum += v
    return [xmin, ymin, xmax, ymax], vsum / len(pts)


key_points = ["nose",
              "left_eye", "right_eye",
              "left_ear", "right_ear",

              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",

              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

anno_path = '../data/hico/train_GT_HICO_with_pose.pkl'
with open(anno_path) as f:
    anno_db = pickle.load(f)

img_path_template = '../data/hico/images/train2015/HICO_train2015_%s.jpg'
for ins_anno in anno_db:
    img_id = ins_anno[0]
    img_path = img_path_template % (str(img_id).zfill(8))
    raw_kps = ins_anno[5]
    boxes = []
    confs = ['head: %.2f', 'left: %.2f', 'right: %.2f', 'foot: %.2f']
    if raw_kps is not None and len(raw_kps) == 51:
        raw_kps = np.reshape(raw_kps, (17, 3))
        head_kps = raw_kps[:5]
        foot_kps = raw_kps[11:]
        left_hand_kps = raw_kps[[5,7,9]]
        right_hand_kps = raw_kps[[6,8,10]]
        kp_groups = [head_kps, left_hand_kps, right_hand_kps, foot_kps]
        for i, kp_group in enumerate(kp_groups):
            box, conf = get_box(kp_group)
            boxes.append(box)
            confs[i] = confs[i] % conf

    show_boxes(img_path, boxes, confs)

