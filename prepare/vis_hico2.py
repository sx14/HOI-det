import pickle
from utils.show_box import show_boxes

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
    kps = []
    confs = []
    if raw_kps is not None and len(raw_kps) == 51:
        for i in range(17):

            x = raw_kps[i * 3 + 0]
            y = raw_kps[i * 3 + 1]
            v = raw_kps[i * 3 + 2]
            kps.append([
                x - 5,
                y - 5,
                x + 5,
                y + 5
            ])
            confs.append(key_points[i])
    show_boxes(img_path, kps, confs)

