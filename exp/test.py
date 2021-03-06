import os
import yaml
import pickle

from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_data import prepare_hico, load_hoi_classes
from dataset import HICODatasetSpa, gen_spatial_map, gen_pose_feat
from model import SpaLan
from load_data import load_image_info, extract_spatial_feature, object_class_mapping
from generate_HICO_detection import generate_HICO_detection


def test_image(model, im_obj_dets, image_size, det_obj2hoi_obj, obj2vec):
    # save image information
    results = []
    spa_maps = torch.FloatTensor(1)
    spa_maps = Variable(spa_maps).cuda()

    obj_class_num = obj2vec.shape[0]
    obj_vecs = torch.zeros((1, obj_class_num))
    obj_vecs = Variable(obj_vecs).cuda()

    pose_vecs = torch.zeros((1, 17))
    pose_vecs = Variable(pose_vecs).cuda()

    hum_thr = 0.8
    obj_thr = 0.3

    for hum_det in im_obj_dets:
        if (np.max(hum_det[5]) > hum_thr) and (hum_det[1] == 'Human'):
            # This is a valid human
            hbox = hum_det[2]
            hscore = hum_det[5]
            skeleton = hum_det[6]
            for obj_det in im_obj_dets:
                if (np.max(obj_det[5]) > obj_thr) and not (np.all(obj_det[2] == hum_det[2])):
                    # This is a valid object
                    obox = obj_det[2]
                    oscore = obj_det[5]
                    oind = det_obj2hoi_obj[obj_det[4]]

                    spa_map_raw = gen_spatial_map(hbox, obox, oind, 80)
                    spa_map_raw = torch.from_numpy(spa_map_raw[np.newaxis, :, :, :])

                    pose_feat_raw = gen_pose_feat(skeleton, obox)
                    pose_feat_raw = torch.from_numpy(pose_feat_raw[np.newaxis, :, :, :])

                    # ovec = torch.from_numpy(obj2vec[oind]).view((1, -1))
                    # obj_vecs.data.resize_(ovec.size()).copy_(ovec)
                    obj_vecs[0, oind] = 1
                    spa_maps.data.resize_(spa_map_raw.size()).copy_(spa_map_raw)
                    pose_feat_raw.data.resize_(pose_feat_raw.size()).copy_(pose_feat_raw)

                    with torch.no_grad():
                        bin_prob, hoi_prob, _, _, _, _ = model(spa_maps, obj_vecs, pose_vecs)

                    temp = []
                    temp.append(hum_det[2])             # Human box
                    temp.append(obj_det[2])             # Object box
                    temp.append(obj_det[4])             # Object class
                    temp.append(hoi_prob.cpu().data.numpy()[0].tolist())            # Score (600)
                    temp.append(hscore)                 # Human score
                    temp.append(oscore)                 # Object score
                    temp.append(bin_prob.cpu().data.numpy()[0].tolist())            # binary score
                    results.append(temp)

    return results


def test(model, obj_det_db, image_set_info, det_obj2hoi_obj, obj2vec):
    all_results = {}
    num_im = len(obj_det_db)
    for i, im_id in enumerate(obj_det_db):
        print('test [%d/%d]' % (i+1, num_im))
        image_size = image_set_info[im_id]
        im_obj_dets = obj_det_db[im_id]
        im_hoi_dets = test_image(model, im_obj_dets, image_size, det_obj2hoi_obj, obj2vec)
        all_results[im_id] = im_hoi_dets
    return all_results


if __name__ == '__main__':
    config_path = 'hico_spa.yaml'
    with open(config_path) as f:
        config = yaml.load(f)

    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, 'all_hoi_detections.pkl')
    if not os.path.exists(output_path):
        print('Loading models ...')
        model_save_dir = config['model_save_dir']
        model = SpaLan(config['spa_feature_dim'],
                       config['num_hoi_classes'],
                       config['num_obj_classes'],
                       config['num_key_points'])
        model = model.cuda()
        resume_dict = torch.load(os.path.join(model_save_dir, '%s_99_weights.pkl' % model))
        model.load_state_dict(resume_dict)
        model.eval()

        data_root = '../data/hico'
        hoi_classes_path = os.path.join(data_root, 'hoi_categories.pkl')
        hoi_classes, obj_classes, vrb_classes, hoi2int = load_hoi_classes(hoi_classes_path)
        det_obj2hoi_obj = object_class_mapping(hoi_classes, obj_classes)
        image_set_info = load_image_info(os.path.join(data_root, 'anno_bbox_full.mat'),
                                         config['data_save_dir'], image_set='test')
        obj2vec_path = os.path.join(config['data_save_dir'], 'hico_obj2vec.pkl')
        with open(obj2vec_path) as f:
            obj2vec = pickle.load(f)

        print('Loading object detections ...')
        obj_det_path = os.path.join(data_root, 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl')
        with open(obj_det_path) as f:
            obj_det_db = pickle.load(f)

        print('Testing ...')
        all_results = test(model, obj_det_db, image_set_info, det_obj2hoi_obj, obj2vec)

        print('Saving results ...')
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        print('Done.')

    generate_HICO_detection(output_path, 'output/results', 1.0, 0)

    os.chdir('benchmark')
    os.system('matlab -nodesktop -nosplash -r "Generate_detection '+ '../output/results/' + '/;quit;"')
