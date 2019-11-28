import os
import numpy as np
from tqdm import tqdm

depth_root = '/home/iznauy/sunx-workspace/datasets/hico_20160224_det/depths/test2015/'
depth_root1 = '/home/iznauy/sunx-workspace/datasets/hico_20160224_det/depths1/test2015/'

for depth_id in tqdm(os.listdir(depth_root)):
    dp = np.load(os.path.join(depth_root, depth_id))
    dp = dp[0][0]
    dp = dp[:, :, np.newaxis]
    new_path = os.path.join(depth_root1, depth_id[:-9]+'.npy')
    np.save(new_path, dp)
