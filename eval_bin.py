import os
import h5py
import json
import pickle
import numpy as np
from tqdm import tqdm

COCO_CLASSES = (
    'background',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')

def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area



def compute_iou(bbox1, bbox2, verbose=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou


def box_label_recall(gt_hois, human_boxes, object_boxes, object_labels, iou_thresh, hoi_list):
    num_pred_human_boxes = len(human_boxes)
    num_pred_object_boxes = len(object_boxes)
    num_pred_connections = num_pred_human_boxes * num_pred_object_boxes

    hoi_dict = {hoi['id']: hoi for hoi in hoi_list}

    num_gt_connections_recalled = 0
    num_gt_connections = 0
    num_gt_human_boxes_recalled = 0
    num_gt_human_boxes = 0
    num_gt_object_boxes_recalled = 0
    num_gt_object_boxes = 0

    for hois_per_type in gt_hois:
        gt_id = hois_per_type['id']
        gt_hoi = hoi_dict[gt_id]

        gt_connections = hois_per_type['connections']
        gt_human_boxes = hois_per_type['human_bboxes']
        gt_object_boxes = hois_per_type['object_bboxes']
        invis = hois_per_type['invis']

        gt_human_boxes_recalled = [False] * len(gt_human_boxes)
        for i, gt_box in enumerate(gt_human_boxes):
            for box in human_boxes:
                try:
                    iou = compute_iou(box, gt_box)
                except:
                    import pdb;
                    pdb.set_trace()
                if iou >= iou_thresh:
                    gt_human_boxes_recalled[i] = True
                    break

        gt_object_boxes_recalled = [False] * len(gt_object_boxes)
        for i, gt_box in enumerate(gt_object_boxes):
            for box, label in zip(object_boxes, object_labels):
                try:
                    iou = compute_iou(box, gt_box)
                except:
                    import pdb;
                    pdb.set_trace()
                if iou >= iou_thresh and label == gt_hoi['object']:
                    gt_object_boxes_recalled[i] = True
                    break

        gt_connections_recalled = [False] * len(gt_connections)
        for k, (i, j) in enumerate(gt_connections):
            if gt_human_boxes_recalled[i] and gt_object_boxes_recalled[j]:
                gt_connections_recalled[k] = True

        num_gt_connections += len(gt_connections)
        num_gt_connections_recalled += gt_connections_recalled.count(True)

        num_gt_human_boxes += len(gt_human_boxes)
        num_gt_human_boxes_recalled += gt_human_boxes_recalled.count(True)

        num_gt_object_boxes += len(gt_object_boxes)
        num_gt_object_boxes_recalled += gt_object_boxes_recalled.count(True)

    try:
        connection_recall = num_gt_connections_recalled / num_gt_connections
    except ZeroDivisionError:
        connection_recall = None

    try:
        human_recall = num_gt_human_boxes_recalled / num_gt_human_boxes
    except ZeroDivisionError:
        human_recall = None

    try:
        object_recall = num_gt_object_boxes_recalled / num_gt_object_boxes
    except ZeroDivisionError:
        object_recall = None

    stats = {
        'connection_recall': connection_recall,
        'human_recall': human_recall,
        'object_recall': object_recall,
        'num_gt_connections_recalled': num_gt_connections_recalled,
        'num_gt_connections': num_gt_connections,
        'num_gt_human_boxes_recalled': num_gt_human_boxes_recalled,
        'num_gt_human_boxes': num_gt_human_boxes,
        'num_gt_object_boxes_recalled': num_gt_object_boxes_recalled,
        'num_gt_object_boxes': num_gt_object_boxes,
        'num_connection_proposals': num_pred_connections,
        'num_human_proposals': num_pred_human_boxes,
        'num_object_proposals': num_pred_object_boxes,
    }

    return stats


def evaluate_boxes_and_labels(select_boxes, output_root, data_root):

    print('Loading anno_list.json ...')
    anno_list_path = os.path.join(data_root, 'anno_list.json')
    with open(anno_list_path) as f:
        anno_list = json.load(f)

    print('Loading hoi_list.json ...')
    hoi_list_path = os.path.join(data_root, 'hoi_list.json')
    with open(hoi_list_path) as f:
        hoi_list = json.load(f)

    print('Evaluating box proposals ...')
    evaluation_stats = {
        'num_gt_connections_recalled': 0,
        'num_gt_connections': 0,
        'num_gt_human_boxes_recalled': 0,
        'num_gt_human_boxes': 0,
        'num_gt_object_boxes_recalled': 0,
        'num_gt_object_boxes': 0,
        'num_connection_proposals': 0,
        'num_human_proposals': 0,
        'num_object_proposals': 0,
    }

    index_error_misses = 0
    num_images = 0
    for anno in tqdm(anno_list):
        global_id = anno['global_id']
        if 'test' in global_id:
            num_images += 1
        else:
            continue

        im_proposals = select_boxes[global_id]

        if len(im_proposals['human_boxes']) == 0:
            human_boxes = np.zeros((0, 4))
            human_scores = np.zeros(0)
            object_boxes = np.zeros((0, 4))
            object_scores = np.zeros(0)
            object_labels = np.zeros(0)
            inter_scores = np.zeros(0)
        else:
            human_boxes = np.array(im_proposals['human_boxes'])[:, :4]
            human_scores = np.array(im_proposals['human_boxes'])[:, 4]
            object_boxes = np.array(im_proposals['object_boxes'])[:, :4]
            object_scores = np.array(im_proposals['object_boxes'])[:, 4]
            object_labels = np.array(im_proposals['object_labels'])
            inter_scores = np.array(im_proposals['interactiveness'])

        good_inds = (human_scores > 0.4) & (object_scores > 0.4) & (inter_scores > 0.3)

        try:
            recall_stats = box_label_recall(
                anno['hois'],
                human_boxes[good_inds].tolist(),
                object_boxes[good_inds].tolist(),
                object_labels[good_inds],
                0.5,
                hoi_list)
        except IndexError:
            index_error_misses += 1
            num_images -= index_error_misses

        for k in evaluation_stats.keys():
            evaluation_stats[k] += recall_stats[k]

    evaluation_stats['human_recall'] = \
        evaluation_stats['num_gt_human_boxes_recalled'] / \
        evaluation_stats['num_gt_human_boxes']
    evaluation_stats['object_recall'] = \
        evaluation_stats['num_gt_object_boxes_recalled'] / \
        evaluation_stats['num_gt_object_boxes']
    evaluation_stats['connection_recall'] = \
        evaluation_stats['num_gt_connections_recalled'] / \
        evaluation_stats['num_gt_connections']
    evaluation_stats['average_human_proposals_per_image'] = \
        evaluation_stats['num_human_proposals'] / num_images
    evaluation_stats['average_object_proposals_per_image'] = \
        evaluation_stats['num_object_proposals'] / num_images
    evaluation_stats['average_connection_proposals_per_image'] = \
        evaluation_stats['average_human_proposals_per_image'] * \
        evaluation_stats['average_object_proposals_per_image']
    evaluation_stats['index_error_misses'] = index_error_misses

    evaluation_stats_json = os.path.join(
        output_root,
        'eval_stats_boxes_labels.json')

    with open(evaluation_stats_json, 'w') as f:
        json.dump(evaluation_stats, f, indent=4)