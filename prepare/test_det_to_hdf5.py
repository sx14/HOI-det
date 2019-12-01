import pickle
import h5py
from tqdm import tqdm
import numpy as np

import _init_paths
from datasets.hico2 import hico2

hoi_classes, obj_classes, vrb_classes, obj2int, hoi2vrb, vrb2hoi = hico2.load_hoi_classes('data/hico')

hoi_ranges = [(161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576), (32, 46),
              (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129), (130, 146),
              (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584), (353, 356),
              (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), (233, 235),
              (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305), (331, 336), (377, 383), (484, 488),
              (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), (258, 264), (274, 283), (357, 363), (419, 429),
              (306, 313), (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), (108, 111), (551, 558), (195, 198),
              (384, 389), (394, 397), (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), (547, 550), (450, 453),
              (430, 434), (248, 252), (291, 295), (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]

org_ind2obj = [-1] + [hoi_classes[hoi_range[0]-1].object_name().replace('_', ' ') for hoi_range in hoi_ranges]


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

coco_obj2ind = dict(zip(COCO_CLASSES, range(len(COCO_CLASSES))))


def load_test_object_detections(det_path):
    print('Loading object detections ...')
    with open(det_path) as f:
        det_db = pickle.load(f)
    return det_db


def to_h5_format(det_db, save_path):
    print('Converting to HDF5 ...')
    det_db_h5 = h5py.File(save_path, 'w')
    image_id_template = 'HICO_test2015_%s'
    for i, im_id in tqdm(enumerate(det_db)):
        image_id = image_id_template % str(im_id).zfill(8)
        det_db_h5.create_group(image_id)

        coco_cls2dets = {coco_cls: [] for coco_cls in COCO_CLASSES}

        image_dets = det_db[im_id]
        for det_id in range(len(image_dets)):
            raw_det = image_dets[det_id]
            raw_cls = raw_det[4]
            coco_cls = org_ind2obj[raw_cls]
            det = [raw_det[2][0], raw_det[2][1],
                   raw_det[2][2], raw_det[2][3],
                   raw_det[5], det_id]
            if det[2] > det[0] and det[3] > det[1]:
                coco_cls2dets[coco_cls].append(det)
            else:
                print('one invalid box !!!')

        image_box_score_ids = []
        image_start_end_ids = []
        start_id = 0
        for coco_ind, coco_cls in enumerate(COCO_CLASSES):
            cls_dets = coco_cls2dets[coco_cls]

            if len(cls_dets) == 0:
                cls_dets.append([0, 0, 10, 10, 0, 0])

            end_id = start_id + len(cls_dets)
            image_box_score_ids += cls_dets
            image_start_end_ids.append([start_id, end_id])
            start_id = end_id

        det_db_h5[image_id].create_dataset('boxes_scores_rpn_ids', data=np.array(image_box_score_ids))
        det_db_h5[image_id].create_dataset('start_end_ids', data=np.array(image_start_end_ids))
    det_db_h5.close()


if __name__ == '__main__':
    det_path = 'data/hico/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl'
    det_db = load_test_object_detections(det_path)
    to_h5_format(det_db, 'selected_coco_cls_dets_test.hdf5')

