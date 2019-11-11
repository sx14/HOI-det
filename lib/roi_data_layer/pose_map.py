import pickle
import numpy as np

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
        'head': 0.1,
        'left_hand': 0.1,
        'right_hand': 0.1,
        'hip': 0.1,
        'left_leg': 0.2,
        'right_leg': 0.2
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


def gen_pose_obj_map(hbox, obox, ibox, skeleton):
    h_xmin, h_ymin, h_xmax, h_ymax = hbox
    o_xmin, o_ymin, o_xmax, o_ymax = obox
    i_xmin, i_ymin, i_xmax, i_ymax = ibox

    human_wh = [h_xmax - h_xmin + 1, h_ymax - h_ymin + 1]
    interact_wh = [i_xmax - i_xmin + 1, i_ymax - i_ymin + 1]

    skeleton[:, 0] = skeleton[:, 0] - i_xmin
    skeleton[:, 1] = skeleton[:, 1] - i_ymin

    x_ratio = 64.0 / interact_wh[0]
    y_ratio = 64.0 / interact_wh[1]

    pose_obj_map = np.zeros((7, 64, 64))
    for i, body_part in enumerate(body_parts):
        box_conf = gen_body_part_box(skeleton, human_wh, body_part)
        if box_conf is not None:
            xmin, ymin, xmax, ymax, conf = box_conf
            xmin = int(xmin * x_ratio)
            ymin = int(ymin * y_ratio)
            xmax = int(xmax * x_ratio)
            ymax = int(ymax * y_ratio)
            pose_obj_map[i, ymin:ymax+1, xmin:xmax+1] = conf
    o_xmin = o_xmin * x_ratio
    o_ymin = o_ymin * y_ratio
    o_xmax = o_xmax * x_ratio
    o_ymax = o_ymax * y_ratio
    pose_obj_map[6, o_ymin:o_ymax+1, o_xmin:o_xmax+1] = 1
    return pose_obj_map