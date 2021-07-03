
# @CreateTime : 2021/3/24
# @Author : sunx

import os
import pickle
import numpy as np
import shutil
from collections import defaultdict
from utils import *
from scipy.misc import imread
import cv2
from tqdm import tqdm

data_root = 'data/hoia/train'
# for image_name in tqdm(os.listdir(data_root)):
#     image_path = os.path.join(data_root, image_name)
#     image1 = imread(image_path)
#     image2 = cv2.imread(image_path)
#     if len(image1.shape) != 3 or image1.shape[2] != 3:
#         print(image_name)

image_path = os.path.join(data_root, 'trainval_028805.jpg')
image1 = imread(image_path)
image2 = cv2.imread(image_path)
a = 1
