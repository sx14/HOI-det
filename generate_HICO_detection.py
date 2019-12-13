# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Change the HICO-DET detection results to the right format.
input arg: python Generate_HICO_detection_nis.py (1:pkl_path) (2:hico_dir) (3:rule_inter) (4:threshold_x) (5:threshold_y)
"""

import pickle
import shutil
import numpy as np
import scipy.io as sio
import os
import sys
import matplotlib
import matplotlib.pyplot as plth
import random

# all the no-interaction HOI index in HICO dataset
hoi_no_inter_all = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186, 194, 198, 208, 214,
                    224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290, 295, 305, 313, 325, 330, 336, 342, 348,
                    352, 356, 363, 368, 376, 383, 389, 393, 397, 407, 414, 418, 429, 434, 438, 445, 449, 453, 463, 474,
                    483, 488, 502, 506, 516, 528, 533, 538, 546, 550, 558, 562, 567, 576, 584, 588, 595, 600]
# all HOI index range corresponding to different object id in HICO dataset
hoi_ranges = [(161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576), (32, 46),
              (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129), (130, 146),
              (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584), (353, 356),
              (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), (233, 235),
              (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305), (331, 336), (377, 383), (484, 488),
              (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), (258, 264), (274, 283), (357, 363), (419, 429),
              (306, 313), (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), (108, 111), (551, 558), (195, 198),
              (384, 389), (394, 397), (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), (547, 550), (450, 453),
              (430, 434), (248, 252), (291, 295), (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]

# object(1 base)
org_obj2hoi = [-1] + [hoi_range[0] - 1 for hoi_range in hoi_ranges]


pair_total_num = 999999

pair_is_del = np.zeros(pair_total_num, dtype='float32')
pair_in_the_result = np.zeros(9999, dtype='float32')


def getSigmoid(b, c, d, x, a=6):
    e = 2.718281828459
    return a / (1 + e ** (b - c * x)) + d


def save_HICO(HICO, HICO_dir, thres_inter, classid, begin, finish):
    # if class is "dog"
    # only consider: "watch_dog", "walk_dog", ..., "no_interact_dog"

    all_boxes = []
    possible_hoi_range = hoi_ranges[classid - 1]
    num_delete_pair_a = 0
    num_delete_pair_b = 0
    num_delete_pair_c = 0

    for i in range(finish - begin + 1):  # for every verb, iteration all the pkl file
        total = []
        score = []

        for key, value in HICO.iteritems():
            for element in value:
                if element[2] == classid:
                    temp = []
                    if isinstance(element[0], list):
                        temp.append(element[0])
                    else:
                        temp.append(element[0].tolist())  # Human box
                    if isinstance(element[1], list):
                        temp.append(element[1])
                    else:
                        temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))    # action id (0-599)

                    human_score = element[4]
                    object_score = element[5]

                    interactiveness = element[6][0]

                    score_old = element[3][begin - 1 + i] * human_score * object_score
                    print 'H: %.4f O: %.4f I: %.4f' % (human_score, object_score, element[3][begin - 1 + i])

                    hoi_num = begin - 1 + i

                    score_new = score_old

                    if (interactiveness < thres_inter):
                        # 1. Non-interactiveness is great enough
                        # 2. Current image contains HOI instances

                        if not ((hoi_num + 1) in hoi_no_inter_all):
                            # Current HOI class is not "no_interaction".
                            # Skip the 520 interactive classes.
                            continue

                    temp.append(score_new)
                    total.append(temp)
                    score.append(score_new)

        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])

    # save the detection result in .mat file
    savefile = os.path.join(HICO_dir, 'detections_' + str(classid).zfill(2) + '.mat')
    if os.path.exists(savefile):
        os.remove(savefile)
    sio.savemat(savefile, {'all_boxes': all_boxes})

    print('class', classid, 'finished')
    num_delete_inter = num_delete_pair_a + num_delete_pair_b

    return num_delete_inter, num_delete_pair_c


def generate_HICO_detection(HICO, HICO_dir, thres_inter):
    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)
    # del_i and del_ni

    del_i = 0
    del_ni = 0

    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 1, 161, 170)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 1 person
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 2, 11, 24)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 2 bicycle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 3, 66, 76)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 3 car
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 4, 147, 160)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 4 motorcycle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 5, 1, 10)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 5 airplane
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 6, 55, 65)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 6 bus
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 7, 187, 194)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 7 train
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 8, 568, 576)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 8 truck
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 9, 32, 46)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 9 boat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 10, 563, 567)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 10 traffic light
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 11, 326, 330)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 11 fire_hydrant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 12, 503, 506)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 12 stop_sign
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 13, 415, 418)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 13 parking_meter
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 14, 244, 247)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 14 bench
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 15, 25, 31)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 15 bird
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 16, 77, 86)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 16 cat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 17, 112, 129)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 17 dog
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 18, 130, 146)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 18 horse
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 19, 175, 186)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 19 sheep
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 20, 97, 107)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 20 cow
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 21, 314, 325)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 21 elephant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 22, 236, 239)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 22 bear
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 23, 596, 600)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 23 zebra
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 24, 343, 348)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 24 giraffe
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 25, 209, 214)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 25 backpack
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 26, 577, 584)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 26 umbrella
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 27, 353, 356)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 27 handbag
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 28, 539, 546)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 28 tie
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 29, 507, 516)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 29 suitcase
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 30, 337, 342)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 30 Frisbee
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 31, 464, 474)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 31 skis
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 32, 475, 483)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 32 snowboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 33, 489, 502)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 33 sports_ball
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 34, 369, 376)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 34 kite
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 35, 225, 232)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 35 baseball_bat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 36, 233, 235)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 36 baseball_glove
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 37, 454, 463)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 37 skateboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 38, 517, 528)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 38 surfboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 39, 534, 538)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 39 tennis_racket
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 40, 47, 54)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 40 bottle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 41, 589, 595)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 41 wine_glass
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 42, 296, 305)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 42 cup
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 43, 331, 336)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 43 fork
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 44, 377, 383)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 44 knife
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 45, 484, 488)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 45 spoon
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 46, 253, 257)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 46 bowl
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 47, 215, 224)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 47 banana
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 48, 199, 208)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 48 apple
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 49, 439, 445)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 49 sandwich
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 50, 398, 407)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 50 orange
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 51, 258, 264)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 51 broccoli
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 52, 274, 283)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 52 carrot
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 53, 357, 363)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 53 hot_dog
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 54, 419, 429)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 54 pizza
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 55, 306, 313)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 55 donut
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 56, 265, 273)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 56 cake
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 57, 87, 92)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 57 chair
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 58, 93, 96)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 58 couch
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 59, 171, 174)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 59 potted_plant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 60, 240, 243)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 60 bed
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 61, 108, 111)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 61 dining_table
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 62, 551, 558)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 62 toilet
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 63, 195, 198)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 63 TV
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 64, 384, 389)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 64 laptop
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 65, 394, 397)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 65 mouse
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 66, 435, 438)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 66 remote
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 67, 364, 368)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 67 keyboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 68, 284, 290)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 68 cell_phone
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 69, 390, 393)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 69 microwave
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 70, 408, 414)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 70 oven
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 71, 547, 550)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 71 toaster
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 72, 450, 453)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 72 sink
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 73, 430, 434)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 73 refrigerator
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 74, 248, 252)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 74 book
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 75, 291, 295)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 75 clock
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 76, 585, 588)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 76 vase
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 77, 446, 449)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 77 scissors
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 78, 529, 533)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 78 teddy_bear
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 79, 349, 352)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 79 hair_drier
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_inter, 80, 559, 562)
    del_i += num_del_i
    del_ni += num_del_no_i
    # 80 toothbrush

    print('num_del_inter', del_i, 'num_del_no_inter', del_ni)

