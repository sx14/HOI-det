# @CreateTime : 2021/3/9
# @Author : sunx

import os
import random
import cv2
import json
import numpy as np

def random_color(n=100):
    colors = []
    for i in range(n):
        r = random.randint(0, 256)
        g = random.randint(0, 256)
        b = random.randint(0, 256)
        colors.append([r, g, b])
    return colors


def show_boxes(im, boxes, labels):
    box_num = len(boxes)
    colors = random_color(box_num)
    im_show = np.copy(im)
    for i in range(box_num):
        box = boxes[i]
        label = labels[i]
        cv2.rectangle(im_show, box[0:2], box[2:4], colors[i], 2)
        cv2.putText(im_show, '%s' % label, (box[0], box[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
        cv2.imshow('123', im_show)
        cv2.waitKey(0)
    return im_show

def load_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
    print('saved at %s' % path)

def load_image_list(path):
    with open(path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    return image_list
