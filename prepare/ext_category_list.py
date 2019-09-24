import pickle

import _init_paths
from datasets.hico import hico
from utils.show_box import show_boxes

hico_ds = hico('train', 'full')
hico_roidb = hico_ds.gt_roidb()
hico_verb_classes = hico_ds.verb_classes
hico_object_classes = hico_ds.object_classes
hico_hoi_classes = hico_ds.hoi_classes

hico_hoi_class_names = [hico_hoi_class.hoi_name() for hico_hoi_class in hico_hoi_classes]
pickle.dump(hico_hoi_class_names, open('HICO_hoi_categories.pkl', 'wb'))


