import os
import json
import pickle

from utils.show_box import show_boxes


def load_verbs(verb2index_path):
    with open(verb2index_path) as f:
        vrb2ind = json.load(f)
        vrb_classes = [0] * len(vrb2ind)
        for vrb, ind in vrb2ind.items():
            vrb_classes[ind] = vrb
    return vrb_classes


def load_objects(object2index_path):
    with open(object2index_path) as f:
        obj2ind = json.load(f)
        obj_classes = [0] * len(obj2ind)
        for obj, ind in obj2ind.items():
            obj_classes[ind] = obj
    return obj_classes


def load_objects1(object_list_path):
    with open(object_list_path) as f:
        obj_classes = f.readlines()
    return obj_classes


def show_positive_instances(image_root, image_template, anno_path, obj_classes, vrb_classes):
    print('Loading annotations ...')
    with open(anno_path) as f:
        annos = pickle.load(f)

    for anno in annos:
        im_id = anno[0]
        im_path = os.path.join(image_root,
                               image_template % str(im_id).zfill(12))
        hbox = anno[2]
        obox = anno[3]

        vrb_inds = anno[1]
        obj_ind = anno[6]
        vrbs = ','.join([vrb_classes[vrb_ind] for vrb_ind in vrb_inds])
        obj = obj_classes[obj_ind]

        show_boxes(im_path, [hbox, obox], [vrbs, obj], ['red', 'blue'])


def show_negative_instances(image_root, image_template, anno_path, obj_classes, vrb_classes):
    print('Loading annotations ...')
    with open(anno_path) as f:
        annos = pickle.load(f)

    for im_id in annos:
        im_path = os.path.join(image_root,
                               image_template % str(im_id).zfill(12))
        im_annos = annos[im_id]
        for anno in im_annos:
            hbox = anno[2]
            obox = anno[3]

            obj_ind = anno[5]
            vrb = 'no_action'
            obj = obj_classes[obj_ind]

            show_boxes(im_path, [hbox, obox], [vrb, obj], ['red', 'blue'])

if __name__ == '__main__':
    vrb2ind_path = '/home/magus/dataset3/coco2014/action_index.json'
    obj2ind_path = '/home/magus/dataset3/coco2014/crehuxw/our_coco_object_classes.json'
    obj_list_path = '/home/magus/dataset3/coco2014/object_index.txt'
    vrb_classes = load_verbs(vrb2ind_path)
    # obj_classes = load_objects(obj2ind_path)
    obj_classes = load_objects1(obj_list_path)

    image_root = '/home/magus/dataset3/coco2014/train2014'
    image_template = 'COCO_train2014_%s.jpg'

    # anno_path = '/home/magus/dataset3/coco2014/crehuxw/Trainval_GT3_VCOCO_with_pose.pkl'
    # show_positive_instances(image_root, image_template, anno_path, obj_classes, vrb_classes)

    anno_path = '/home/magus/dataset3/coco2014/crehuxw/Trainval_Neg3_VCOCO_with_pose.pkl'
    show_negative_instances(image_root, image_template, anno_path, obj_classes, vrb_classes)