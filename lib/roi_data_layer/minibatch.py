# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
from random import randint




def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, dp_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'image': im_blob,
           'depth': dp_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  # boxes: (x1, y1, x2, y2)
  hboxes = roidb[0]['hboxes'] * im_scales[0]
  oboxes = roidb[0]['oboxes'] * im_scales[0]
  iboxes = roidb[0]['iboxes'] * im_scales[0]
  pboxes = roidb[0]['pbox_lists'].copy()
  pboxes[:, :, :4] = pboxes[:, :, :4] * im_scales[0]
  pboxes1 = roidb[0]['pbox_lists1'].copy()
  pboxes1[:, :, :4] = pboxes1[:, :, :4] * im_scales[0]

  hoi_classes = roidb[0]['hoi_classes']
  vrb_classes = roidb[0]['vrb_classes']
  obj_classes = roidb[0]['obj_classes']
  bin_classes = roidb[0]['bin_classes']
  hoi_masks = roidb[0]['hoi_masks']
  vrb_masks = roidb[0]['vrb_masks']

  blobs['hboxes'] = hboxes
  blobs['oboxes'] = oboxes
  blobs['iboxes'] = iboxes
  blobs['pbox_lists'] = pboxes
  blobs['pbox_lists1'] = pboxes1
  blobs['hoi_classes'] = hoi_classes
  blobs['vrb_classes'] = vrb_classes
  blobs['obj_classes'] = obj_classes
  blobs['bin_classes'] = bin_classes
  blobs['hoi_masks'] = hoi_masks
  blobs['vrb_masks'] = vrb_masks
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
  blobs['img_id'] = roidb[0]['img_id']

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  im_scales = []
  processed_ims = []
  processed_dps = []

  for i in range(num_images):
    # print(roidb[i]['image'])
    im = imread(roidb[i]['image'])
    # dp = np.load(roidb[i]['depth'])
    # dp = np.zeros((im.shape[0], im.shape[1], 7))

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)

    if im.shape[2] > 3:
      im = im[:, :, :3]

    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      # dp = dp[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    # dp, de_scale = prep_im_for_blob(dp, cfg.DEPTH_MEANS, target_size,
    #                 cfg.TRAIN.MAX_SIZE)

    im_scales.append(im_scale)
    processed_ims.append(im)
    # processed_dps.append(dp)

  # Create a blob to hold the input images
  im_blob = im_list_to_blob(processed_ims, 3)
  # dp_blob = im_list_to_blob(processed_dps, 7)
  dp_blob = None
  return im_blob, dp_blob, im_scales
