import _init_paths
from datasets.hico import hico
from utils.show_box import show_boxes

hico_ds = hico('train', '2016')
hico_roidb = hico_ds.gt_roidb()
hico_verb_classes = hico_ds.verb_classes
hico_object_classes = hico_ds.object_classes

for i, img_rois in enumerate(hico_roidb):
    img_path = hico_ds.image_path_at(i)
    hboxes = img_rois['hboxes']
    hboxes_colors = ['red' for _ in range(len(hboxes))]
    oboxes = img_rois['oboxes']
    oboxes_colors = ['blue' for _ in range(len(oboxes))]
    verb_class_lists = img_rois['verb_classes']
    verb_classes = []
    for verb_class_ids in verb_class_lists:
        verb_names = []
        for verb_class_id in verb_class_ids:
            verb_names.append(hico_verb_classes[verb_class_id])
        verb_classes.append(','.join(verb_names))
    object_class_ids = img_rois['object_classes']
    object_classes = [hico_object_classes[object_class_id] for object_class_id in object_class_ids]

    all_boxes = hboxes.tolist() + oboxes.tolist()
    all_colors = hboxes_colors + oboxes_colors
    all_classes = verb_classes + object_classes

    show_boxes(img_path, all_boxes, all_classes, all_colors)
