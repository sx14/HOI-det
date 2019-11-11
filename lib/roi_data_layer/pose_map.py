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


def est_hand(wrist, elbow):
    return wrist - 0.5 * (wrist - elbow)


def get_body_part_kps(part, all_kps):
    all_part_kps = {
        'left_leg':  ['left_ankle'],
        'right_leg': ['right_ankle'],
        'left_hand':    ['left_hand', 'left_wrist', 'left_elbow'],
        'right_hand':   ['right_hand', 'right_wrist', 'right_elbow'],
        'hip':  ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
        'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    }

    kp2ind = dict(zip(key_points, range(len(key_points))))
    part_kps = np.zeros((len(all_part_kps[part]), 3))
    for i, kp_name in enumerate(all_part_kps[part]):
        if kp_name == 'left_hand':
            left_wrist = all_kps[kp2ind['left_wrist']]
            left_elbow = all_kps[kp2ind['left_elbow']]
            kp = est_hand(left_wrist, left_elbow)
        elif kp_name == 'right_hand':
            right_wrist = all_kps[kp2ind['right_wrist']]
            right_elbow = all_kps[kp2ind['right_elbow']]
            kp = est_hand(right_wrist, right_elbow)
        else:
            kp = all_kps[kp2ind[kp_name]]
        part_kps[i] = kp
    return part_kps


def get_body_part_alpha(part):
    all_body_part_alpha = {
        'head': 0.2,
        'left_hand': 0.2,
        'right_hand': 0.2,
        'hip': 0.25,
        'left_leg': 0.25,
        'right_leg': 0.25
    }
    return all_body_part_alpha[part]


def gen_body_part_box(all_kps, human_wh, part, kp_thr=0.01, area_thr=0):
    part_kps = get_body_part_kps(part, all_kps)
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    conf_sum = 0.0
    for i in range(len(part_kps)):
        conf = part_kps[i, 2]
        if conf < kp_thr:
            return None
        conf_sum += conf
        xmin = min(xmin, part_kps[i, 0])
        ymin = min(ymin, part_kps[i, 1])
        xmax = max(xmax, part_kps[i, 0])
        ymax = max(ymax, part_kps[i, 1])
    conf_avg = conf_sum / len(part_kps)
    if (ymax - ymin + 1) * (xmax - xmin + 1) < area_thr:
        return None
    return [xmin - get_body_part_alpha(part) * human_wh[0],
            ymin - get_body_part_alpha(part) * human_wh[1],
            xmax + get_body_part_alpha(part) * human_wh[0],
            ymax + get_body_part_alpha(part) * human_wh[1],
            conf_avg]


def gen_pose_obj_map(hbox, obox, ibox, skeleton, size=224):
    h_xmin, h_ymin, h_xmax, h_ymax = hbox
    o_xmin, o_ymin, o_xmax, o_ymax = obox
    i_xmin, i_ymin, i_xmax, i_ymax = ibox

    human_wh = [h_xmax - h_xmin + 1, h_ymax - h_ymin + 1]
    interact_wh = [i_xmax - i_xmin + 1, i_ymax - i_ymin + 1]

    skeleton[:, 0] = skeleton[:, 0] - i_xmin
    skeleton[:, 1] = skeleton[:, 1] - i_ymin

    x_ratio = size * 1.0 / interact_wh[0]
    y_ratio = size * 1.0 / interact_wh[1]

    pose_obj_map = np.zeros((8, size, size))
    for i, body_part in enumerate(body_parts):
        box_conf = gen_body_part_box(skeleton, human_wh, body_part)
        if box_conf is not None:
            xmin, ymin, xmax, ymax, conf = box_conf
            xmin = int(xmin * x_ratio)
            ymin = int(ymin * y_ratio)
            xmax = int(xmax * x_ratio)
            ymax = int(ymax * y_ratio)
            pose_obj_map[i, ymin:ymax+1, xmin:xmax+1] = conf

    o_xmin = int((o_xmin - i_xmin) * x_ratio)
    o_ymin = int((o_ymin - i_ymin) * y_ratio)
    o_xmax = int((o_xmax - i_xmin) * x_ratio)
    o_ymax = int((o_ymax - i_ymin) * y_ratio)
    pose_obj_map[6, o_ymin:o_ymax+1, o_xmin:o_xmax+1] = 1

    h_xmin = int((h_xmin - i_xmin) * x_ratio)
    h_ymin = int((h_ymin - i_ymin) * y_ratio)
    h_xmax = int((h_xmax - i_xmin) * x_ratio)
    h_ymax = int((h_ymax - i_ymin) * y_ratio)
    pose_obj_map[7, h_ymin:h_ymax+1, h_xmin:h_xmax+1] = 1
    return pose_obj_map


def show_boxes(im_path, dets, cls=None, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):

        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=1.5)
        )
        if cls is not None and len(cls) == len(dets):
            ax.text(bbox[0], bbox[1] - 2,
                    '{}'.format(cls[i]),
                    bbox=dict(facecolor=colors[i], alpha=0.5),
                    fontsize=14, color='white')
        plt.axis('off')
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    anno_path = '../../data/hico/train_GT_HICO_with_pose.pkl'
    with open(anno_path) as f:
        anno_db = pickle.load(f)

    img_path_template = '../../data/hico/images/train2015/HICO_train2015_%s.jpg'
    for ins_anno in anno_db:
        img_id = ins_anno[0]
        raw_kps = ins_anno[5]
        hbox = ins_anno[2]
        obox = ins_anno[3]
        ibox = [min(hbox[0], obox[0]),
                min(hbox[1], obox[1]),
                max(hbox[2], obox[2]),
                max(hbox[3], obox[3])]
        img_path = img_path_template % (str(img_id).zfill(8))
        if raw_kps is None or len(raw_kps) != 51:
            continue

        all_kps = np.reshape(raw_kps, (len(key_points), 3))
        pose_map = gen_pose_obj_map(hbox, obox, ibox, all_kps, 224)
        im_i0 = cv2.imread(img_path)
        im_i0 = cv2.rectangle(im_i0, (hbox[0], hbox[1]), (hbox[2], hbox[3]), (0, 255, 0), 4)
        im_i0 = cv2.rectangle(im_i0, (obox[0], obox[1]), (obox[2], obox[3]), (0, 0, 255), 4)
        im_i0 = im_i0[ibox[1]:ibox[3]+1, ibox[0]:ibox[2]+1, :]

        for i in range(pose_map.shape[0]):
            im_i = cv2.resize(im_i0, (224, 224))
            channel = pose_map[i]
            im_i[:,:,0][channel > 0] = im_i[:,:,0][channel > 0] / 2
            im_i[:,:,1][channel > 0] = im_i[:,:,1][channel > 0] / 2
            im_i[:,:,2][channel > 0] = im_i[:,:,2][channel > 0] / 2
            im_i = cv2.resize(im_i, (500, 500))
            cv2.imshow('123', im_i)

            cv2.waitKey(0)
