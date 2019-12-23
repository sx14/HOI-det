# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.vcoco import vcoco
from datasets.hico2 import hico2
import numpy as np

# Set up hico_<year>_<split>
for version in ['full', 'mini']:
  for split in ['train', 'test']:
    name = 'hico2_{}_{}'.format(version, split)
    __sets[name] = (lambda split=split, version=version: hico2(split, version))

# Set up hico_<year>_<split>
for version in ['full']:
  for split in ['trainval', 'test']:
    name = 'vcoco_{}_{}'.format(version, split)
    __sets[name] = (lambda split=split, version=version: vcoco(split, version))



def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
