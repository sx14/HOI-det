import pickle
import numpy as np
import cv2

import _init_paths
from generate_HICO_detection import generate_HICO_detection, org_obj2hoi
from datasets.hico2 import hico2


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

key_point_connections = [
    [0, 1],     # nose->left eye
    [0, 2],     # nose->right eye
    [1, 3],     # left eye->left ear
    [2, 4],     # right eye->right ear
    [0, 5],     # nose->left shoulder
    [0, 6],     # nose->right shoulder
    [5, 7],     # left_shoulder->left elbow
    [6, 8],     # right_shoulder->right elbow
    [7, 9],     # left elbow->left wrist
    [8, 10],    # right elbow->right wrist
    [5, 11],    # left shoulder->left hip
    [6, 12],    # right shoulder->right hip
    [11, 13],   # left hip->left knee
    [12, 14],   # right hip->right knee
    [13, 15],   # left knee->left ankle
    [14, 16],   # right knee->right ankle
    [5, 6],     # left shoulder->right shoulder
    [11, 12],   # left hip->right hip
]


def est_hand(wrist, elbow):
    return wrist - 0.5 * (wrist - elbow)


def get_body_part_kps(part, all_kps):
    all_part_kps = {
        'left_leg':  ['left_ankle', 'left_knee'],
        'right_leg': ['right_ankle', 'right_knee'],
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
        'left_leg': 0.1,
        'right_leg': 0.1
    }
    return all_body_part_alpha[part]


def gen_body_part_box(all_kps, human_wh, part):
    part_kps = get_body_part_kps(part, all_kps)
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    for i in range(len(part_kps)):
        xmin = min(xmin, part_kps[i, 0])
        ymin = min(ymin, part_kps[i, 1])
        xmax = max(xmax, part_kps[i, 0])
        ymax = max(ymax, part_kps[i, 1])

    human_scale = (human_wh[0]+human_wh[1]) / 2.0
    return [xmin - get_body_part_alpha(part) * human_scale,
            ymin - get_body_part_alpha(part) * human_scale,
            xmax + get_body_part_alpha(part) * human_scale,
            ymax + get_body_part_alpha(part) * human_scale]

# ========================== gen figures ============================

def draw_human_object(hbox, obox, im_path):
    # figure2: human object (full image)

    im = cv2.imread(im_path)

    # draw human box
    # draw object box
    cv2.rectangle(im, (hbox[0], hbox[1]), (hbox[2], hbox[3]), color=color_h, thickness=ho_thick)
    cv2.rectangle(im, (obox[0], obox[1]), (obox[2], obox[3]), color=color_o, thickness=ho_thick)

    cv2.imshow('123', im)
    cv2.waitKey(0)
    cv2.imwrite('fig2_3.jpg', im)


def draw_human_object_conf_map(hbox, obox, im_path):
    # figure2: human-object configuration

    im = cv2.imread(im_path)
    gr = np.zeros(im.shape).astype(np.uint8)
    gr[:, :] = color_grey
    ubox = [int(min(hbox[0], obox[0])), int(min(hbox[1], obox[1])),
            int(max(hbox[2], obox[2])), int(max(hbox[3], obox[3]))]

    # draw human box
    # draw object box
    cv2.rectangle(gr, (hbox[0], hbox[1]), (hbox[2], hbox[3]), color=color_h, thickness=ho_thick+5)
    cv2.rectangle(gr, (obox[0], obox[1]), (obox[2], obox[3]), color=color_o, thickness=ho_thick+5)

    gr_union = gr[ubox[1]:ubox[3]+1, ubox[0]:ubox[2]+1, :]
    cv2.imshow('123', gr_union)
    cv2.waitKey(0)
    cv2.imwrite('fig2_2.jpg', gr_union)



def draw_human_skeleton_body_part_conf_map(hbox, obox, skeleton, im_path):
    # figure2: body-part configuration

    im = cv2.imread(im_path)
    gr = np.zeros(im.shape).astype(np.uint8)
    gr[:, :] = color_grey
    ubox = [int(min(hbox[0], obox[0])), int(min(hbox[1], obox[1])),
            int(max(hbox[2], obox[2])), int(max(hbox[3], obox[3]))]

    # draw skeleton
    kps = []
    for i in range(skeleton.shape[0]):
        raw_kp = skeleton[i, :].astype(np.int).tolist()
        kps.append((raw_kp[0], raw_kp[1]))

    for kp in kps:
        cv2.circle(gr, kp, kp_thick+2, color_point, -1)

    for kp_cnts in key_point_connections:
        cv2.line(gr, kps[kp_cnts[0]], kps[kp_cnts[1]], color_bone, bone_thick+2)

    # draw body part boxes
    human_wh = [hbox[3] - hbox[1] + 1, hbox[2] - hbox[0] + 1]
    for i, body_part in enumerate(body_parts):
        body_part_box = gen_body_part_box(skeleton, human_wh, body_part)
        body_part_box = [
            int(max(body_part_box[0], ubox[0])),
            int(max(body_part_box[1], ubox[1])),
            int(min(body_part_box[2], ubox[2])),
            int(min(body_part_box[3], ubox[3]))]
        cv2.rectangle(gr, (body_part_box[0], body_part_box[1]), (body_part_box[2], body_part_box[3]),
                      color=color_part, thickness=part_thick+5)

    gr_union = gr[ubox[1]:ubox[3]+1, ubox[0]:ubox[2]+1, :]
    cv2.imshow('123', gr_union)
    cv2.waitKey(0)
    cv2.imwrite('fig2_1.jpg', gr_union)


def draw_human_object_skeleton(hbox, obox, skeleton, im_path):
    # figure2: human-object-skeleton (full image)
    # figure4: human-object-skeleton (union box)

    im = cv2.imread(im_path)
    colors = [color_h, color_o]
    for i, box in enumerate([hbox, obox]):
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), colors[i], thickness=ho_thick)
    kps = []
    for i in range(skeleton.shape[0]):
        raw_kp = skeleton[i, :].astype(np.int).tolist()
        kps.append((raw_kp[0], raw_kp[1]))

    for kp in kps:
        cv2.circle(im, kp, kp_thick, color_point, -1)

    for kp_cnts in key_point_connections:
        cv2.line(im, kps[kp_cnts[0]], kps[kp_cnts[1]], color_bone, bone_thick)

    ubox = [int(min(hbox[0], obox[0])), int(min(hbox[1], obox[1])),
            int(max(hbox[2], obox[2])), int(max(hbox[3], obox[3]))]
    im_union = im[ubox[1]:ubox[3]+1, ubox[0]:ubox[2]+1, :]

    cv2.imwrite('fig4_0.jpg', im_union)
    cv2.imwrite('fig2_0.jpg', im)


def draw_body_parts(hbox, obox, skeleton, im_path):
    # figure 4

    def draw_box_skeleton(box, skeleton, color, im, ubox, output_path):
        # figure 4: box skeleton binary map
        im = im.copy()
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

        kps = []
        for i in range(skeleton.shape[0]):
            raw_kp = skeleton[i, :].astype(np.int).tolist()
            kps.append((raw_kp[0], raw_kp[1]))

        for kp in kps:
            cv2.circle(im, kp, 6, color_point, -1)

        for kp_cnts in key_point_connections:
            cv2.line(im, kps[kp_cnts[0]], kps[kp_cnts[1]], color_bone, 4)

        grey = np.zeros(im.shape).astype(np.uint8)
        grey[:, :] = color_grey
        im_grey = im * 0.3 + grey * 0.7
        grey[box[1]:box[3] + 1, box[0]:box[2] + 1, :] = im_grey[box[1]:box[3] + 1, box[0]:box[2] + 1, :]

        cv2.rectangle(grey, (box[0], box[1]), (box[2], box[3]), color, thickness=6)

        grey_union = grey[ubox[1]:ubox[3] + 1, ubox[0]:ubox[2] + 1, :]
        cv2.imshow('123', grey_union)
        cv2.waitKey(0)
        cv2.imwrite(output_path, grey_union)

    def draw_box(box, color, im, ubox, output_path):
        # figure 4: object binary map

        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        im = im.copy()
        grey = np.zeros(im.shape).astype(np.uint8)
        grey[:, :] = color_grey
        im_grey = im * 0.3 + grey * 0.7
        grey[box[1]:box[3] + 1, box[0]:box[2] + 1, :] = im_grey[box[1]:box[3] + 1, box[0]:box[2] + 1, :]
        cv2.rectangle(grey, (box[0], box[1]), (box[2], box[3]), color, thickness=6)
        grey_union = grey[ubox[1]:ubox[3] + 1, ubox[0]:ubox[2] + 1, :]
        cv2.imshow('123', grey_union)
        cv2.waitKey(0)
        cv2.imwrite(output_path, grey_union)

    im = cv2.imread(im_path)
    ubox = [int(min(hbox[0], obox[0])), int(min(hbox[1], obox[1])),
            int(max(hbox[2], obox[2])), int(max(hbox[3], obox[3]))]

    draw_box(obox, color_o, im, ubox, 'fig4_8.jpg')
    draw_box_skeleton(hbox, skeleton, color_h, im, ubox, 'fig4_7.jpg')

    human_wh = [hbox[3]-hbox[1]+1, hbox[2]-hbox[0]+1]
    for i, body_part in enumerate(body_parts):
        body_part_box = gen_body_part_box(skeleton, human_wh, body_part)
        body_part_box = [
            int(max(body_part_box[0], ubox[0])),
            int(max(body_part_box[1], ubox[1])),
            int(min(body_part_box[2], ubox[2])),
            int(min(body_part_box[3], ubox[3]))]

        draw_box_skeleton(body_part_box, skeleton, color_part, im, ubox, 'fig4_%d.jpg' % (i + 1))


if __name__ == '__main__':

    print('Loading HOI categories ...')
    hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, vrb2hoi = hico2.load_hoi_classes('../data/hico')

    print('Loading detections ...')
    det_path = '../data/hico/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl'
    with open(det_path) as f:
        det_db = pickle.load(f)

    im_id = 9362
    im_dets = det_db[im_id]
    im_path = '../data/hico/images/test2015/HICO_test2015_%s.jpg' % str(im_id).zfill(8)

    hbox = None
    obox = None
    skeleton = None
    for det in im_dets:
        if det[1] == 'Human':
            hbox = det[2]
            raw_kps = det[6]
            skeleton = np.array(raw_kps).reshape((17, 3))
        else:
            obj_name = hoi_classes[org_obj2hoi[det[4]]].object_name()
            if obj_name == 'baseball_bat':
                obox = det[2]

    color_h = (0, 0, 255)
    color_o = (240, 176, 0)
    color_point = (0, 255, 255)
    color_bone = (176, 240, 0)
    color_part = (0, 255, 255)
    color_grey = (207, 207, 207)

    ho_thick = 10
    kp_thick = 6
    part_thick = 6
    bone_thick = 4

    # figure 2
    draw_human_object(hbox, obox, im_path)
    draw_human_object_conf_map(hbox, obox, im_path)
    draw_human_skeleton_body_part_conf_map(hbox, obox, skeleton, im_path)

    # figure 4
    draw_human_object_skeleton(hbox, obox, skeleton, im_path)
    draw_body_parts(hbox, obox, skeleton, im_path)

