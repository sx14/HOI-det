import _init_paths
from datasets.hico2 import hico2
from utils.show_box import show_boxes

hico_ds = hico2('train', 'full')
hico_roidb = hico_ds.gt_roidb()
hico_verb_classes = hico_ds.verb_classes
hico_object_classes = hico_ds.object_classes
hico_hoi_classes = hico_ds.hoi_classes

for obj_class in hico_object_classes:
    print(obj_class)

for i, img_rois in enumerate(hico_roidb):
    img_path = hico_ds.image_path_at(i)
    hboxes = img_rois['hboxes']
    hboxes_colors = ['red' for _ in range(len(hboxes))]
    oboxes = img_rois['oboxes']
    oboxes_colors = ['blue' for _ in range(len(oboxes))]
    hoi_classes = img_rois['hoi_classes']

    verb_classes = []
    for p in range(hoi_classes.shape[0]):
        verb_names = []
        for q in range(hoi_classes.shape[1]):
            if hoi_classes[p, q] == 1:
                verb_names.append(hico_hoi_classes[q].verb_name())
        verb_classes.append(','.join(verb_names))

    object_class_ids = img_rois['obj_classes']
    object_classes = [hico_object_classes[object_class_id] for object_class_id in object_class_ids]

    all_boxes = hboxes.tolist() + oboxes.tolist()
    all_colors = hboxes_colors + oboxes_colors
    all_classes = verb_classes + object_classes

    show_boxes(img_path, all_boxes, all_classes, all_colors)
