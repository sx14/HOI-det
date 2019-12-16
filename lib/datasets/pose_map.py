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

def iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    xmin_i = max(xmin1, xmin2)
    ymin_i = max(ymin1, ymin2)
    xmax_i = min(xmax1, xmax2)
    ymax_i = min(ymax1, ymax2)
    w_i = xmax_i - xmin_i + 1
    h_i = ymax_i - ymin_i + 1
    if w_i > 0 and h_i > 0:
        area_i = w_i * h_i
        return area_i / (area1 + area2 - area_i)
    else:
        return 0



def est_part_boxes(hbox):
    # xmin, ymin, xmax, ymax = hbox
    # width = xmax - xmin + 1
    # height = ymax - ymin + 1
    # if width > height:
    #     p_boxes = []
    #     for i in range(3):
    #         pxmin = xmin + i * width / 3.0
    #         pxmax = xmin + (i+1) * width / 3.0
    #         pymin = ymin
    #         pymax = ymax
    #         p_boxes += [pxmin, pymin, pxmax, pymax]
    #     p_boxes = p_boxes + p_boxes
    # else:
    #     p_boxes = []
    #     for i in range(3):
    #         pxmin = xmin
    #         pxmax = xmax
    #         pymin = ymin + i * height / 3.0
    #         pymax = ymin + (i+1) * height / 3.0
    #         p_boxes += [pxmin, pymin, pxmax, pymax]
    #     p_boxes = p_boxes + p_boxes
    p_boxes = [hbox] * 6
    return p_boxes


def est_hand(wrist, elbow):
    return wrist - 0.5 * (wrist - elbow)


def get_body_part_kps(part, all_kps):
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


def gen_body_part_box(all_kps, human_wh, part):
    part_kps = get_body_part_kps(part, all_kps)
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    conf_sum = 0.0
    for i in range(len(part_kps)):
        xmin = min(xmin, part_kps[i, 0])
        ymin = min(ymin, part_kps[i, 1])
        xmax = max(xmax, part_kps[i, 0])
        ymax = max(ymax, part_kps[i, 1])
        conf_sum += part_kps[i, 2]

    return [xmin - get_body_part_alpha(part) * human_wh[0],
            ymin - get_body_part_alpha(part) * human_wh[1],
            xmax + get_body_part_alpha(part) * human_wh[0],
            ymax + get_body_part_alpha(part) * human_wh[1],
            conf_sum / len(part_kps)]


def gen_part_boxes(hbox, skeleton, im_hw):
    h_xmin, h_ymin, h_xmax, h_ymax = hbox
    h_wh = [h_xmax - h_xmin + 1, h_ymax - h_ymin + 1]

    if skeleton is None:
        return [h_xmin, h_ymin, h_xmax, h_ymax] * len(body_parts)

    part_boxes = []
    for i, body_part in enumerate(body_parts):
        box = gen_body_part_box(skeleton, h_wh, body_part)

        xmin, ymin, xmax, ymax, conf = box
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, im_hw[1]-1)
        ymax = min(ymax, im_hw[0]-1)

        part_boxes.append(xmin)
        part_boxes.append(ymin)
        part_boxes.append(xmax)
        part_boxes.append(ymax)

    return part_boxes


def gen_part_boxes1(hbox, skeleton):
    h_xmin, h_ymin, h_xmax, h_ymax = hbox
    h_wh = [h_xmax - h_xmin + 1, h_ymax - h_ymin + 1]

    if skeleton is None:
        return [h_xmin, h_ymin, h_xmax, h_ymax, 0.0001] * len(body_parts)

    part_boxes = []
    for i, body_part in enumerate(body_parts):
        box = gen_body_part_box(skeleton, h_wh, body_part)
        if iou(box[:4], hbox) == 0:
            part_boxes.append(h_xmin)
            part_boxes.append(h_ymin)
            part_boxes.append(h_xmax)
            part_boxes.append(h_ymax)
        else:
            xmin, ymin, xmax, ymax, conf = box
            xmin = max(h_xmin, xmin)
            ymin = max(h_ymin, ymin)
            xmax = min(h_xmax, xmax)
            ymax = min(h_ymax, ymax)

            part_boxes.append(xmin)
            part_boxes.append(ymin)
            part_boxes.append(xmax)
            part_boxes.append(ymax)
            part_boxes.append(conf)

    return part_boxes


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
        img_path = img_path_template % (str(img_id).zfill(8))
        im_i0 = cv2.imread(img_path)

        raw_kps = ins_anno[5]
        hbox = ins_anno[2]
        obox = ins_anno[3]
        ibox = [min(hbox[0], obox[0]),
                min(hbox[1], obox[1]),
                max(hbox[2], obox[2]),
                max(hbox[3], obox[3])]

        if raw_kps is None or len(raw_kps) != 51:
            part_boxes = est_part_boxes(hbox)
            part_boxes = np.array(part_boxes).reshape((6, 4)).astype(np.int)

            im_i0 = cv2.rectangle(im_i0, (hbox[0], hbox[1]), (hbox[2], hbox[3]), (0, 255, 0), 4)
            im_i0 = cv2.rectangle(im_i0, (obox[0], obox[1]), (obox[2], obox[3]), (0, 0, 255), 4)

        else:
            all_kps = np.reshape(raw_kps, (len(key_points), 3))
            part_boxes = gen_part_boxes(hbox, all_kps, im_i0.shape[:2])
            part_boxes = np.array(part_boxes).reshape((6, 4)).astype(np.int)

            im_i0 = cv2.rectangle(im_i0, (hbox[0], hbox[1]), (hbox[2], hbox[3]), (0, 255, 0), 4)
            im_i0 = cv2.rectangle(im_i0, (obox[0], obox[1]), (obox[2], obox[3]), (0, 0, 255), 4)

        for i in range(part_boxes.shape[0]):
            im_i1 = im_i0.copy()
            im_i1 = cv2.rectangle(im_i1,  (part_boxes[i, 0], part_boxes[i, 1]),
                                          (part_boxes[i, 2], part_boxes[i, 3]), (0, 255, 255), 4)

            cv2.imshow('123', im_i1)
            cv2.waitKey(0)
