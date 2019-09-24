import os
import yaml
import pickle

from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_data import prepare_hico, load_hoi_classes
from dataset import HICODatasetSpa
from model import SpaLan
from load_data import load_image_info, extract_spatial_feature, object_class_mapping
from generate_HICO_detection import generate_HICO_detection


def test_image(model, im_obj_dets, image_size, det_obj2hoi_obj, obj2vec):
    # save image information
    results = []
    in_feat = torch.FloatTensor(1)
    in_feat = Variable(in_feat).cuda()

    hum_thr = 0.8
    obj_thr = 0.3

    for hum_det in im_obj_dets:
        if (np.max(hum_det[5]) > hum_thr) and (hum_det[1] == 'Human'):
            # This is a valid human
            hbox = {
                'xmin': hum_det[2][0],
                'ymin': hum_det[2][1],
                'xmax': hum_det[2][2],
                'ymax': hum_det[2][3],
            }
            hscore = hum_det[5]
            for obj_det in im_obj_dets:
                if (np.max(obj_det[5]) > obj_thr) and not (np.all(obj_det[2] == hum_det[2])):
                    # This is a valid object
                    obox = {
                        'xmin': obj_det[2][0],
                        'ymin': obj_det[2][1],
                        'xmax': obj_det[2][2],
                        'ymax': obj_det[2][3],
                    }
                    oscore = obj_det[5]
                    oind = det_obj2hoi_obj[obj_det[4]]
                    ovec = torch.from_numpy(obj2vec[oind])

                    spa_feat = extract_spatial_feature(hbox, obox, image_size)
                    spa_feat = torch.from_numpy(np.array(spa_feat))
                    in_feat_np = torch.cat([spa_feat, ovec]).reshape((1, -1))
                    in_feat.data.resize_(in_feat_np.size() ).copy_(in_feat_np)
                    with torch.no_grad():
                        bin_prob, hoi_prob, _, _, _, _ = model(in_feat)

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
        model = SpaLan(config['lan_feature_dim'] + config['spa_feature_dim'],
                config['num_classes'])
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
        obj_det_path = os.path.join(data_root, 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl')
        with open(obj_det_path) as f:
            obj_det_db = pickle.load(f)

        print('Testing ...')
        all_results = test(model, obj_det_db, image_set_info, det_obj2hoi_obj, obj2vec)

        print('Saving results ...')
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        print('Done.')

    generate_HICO_detection(output_path, 'output/results', 0.9, 0.1)

    os.chdir('../benchmark_tin')
    os.system('matlab -nodesktop -nosplash -r "Generate_detection '+ 'results/SpaLan/' + '/;quit;"')