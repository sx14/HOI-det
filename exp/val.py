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


def show_scores(hoi_classes, hoi_probs, interval):
    vrb_names = [hoi_classes[i].verb_name() for i in range(interval[0], interval[1] + 1)]
    obj_name = hoi_classes[interval[0]].object_name()

    plt.figure(0)
    plt.title(obj_name)
    plt.bar(vrb_names, hoi_probs[interval[0]:interval[1]+1])
    plt.show()


def val(model, dataset, hoi_classes, hoi2int, show=False):
    error_bin_all = 0.0
    error_hoi_all = 0.0
    batch_count = 0
    ins_count = 0
    print('Evaluating ...')

    for data in dataset:
        batch_count += 1
        if batch_count % 100 == 0:
            print(batch_count)

        spa_maps = Variable(data[0]).cuda()
        obj_vecs = Variable(data[1]).cuda()
        hoi_cates = Variable(data[2]).cuda()
        bin_cates = Variable(data[3]).cuda()

        pos_mask = torch.eq(bin_cates, 0)
        if pos_mask.sum().item() == 0:
            continue

        bin_prob, hoi_prob, \
        loss_bin, loss_hoi, \
        error_bin, error_hoi = model(spa_maps, obj_vecs, hoi_cates, bin_cates, pos_mask)
        num_ins = spa_maps.shape[0]
        error_bin_all += error_bin.data.item()
        error_hoi_all += error_hoi.data.item()
        ins_count += num_ins

        if show:
            for i in range(num_ins):
                gt_hoi_cates = [hoi_classes[ind] for ind, v in enumerate(hoi_cates[i]) if v == 1]
                if bin_cates[i] == 0:
                    gt_pos_neg = 'P'
                    gt_obj_cate_name = gt_hoi_cates[0].object_name()
                    gt_vrb_cate_names = [hoi_cate.verb_name() for hoi_cate in gt_hoi_cates]
                    gt_hoi_cate_str = '%s - ' % gt_obj_cate_name + ','.join(gt_vrb_cate_names)
                else:
                    gt_pos_neg = 'N'
                    gt_hoi_cate_str = ''

                if torch.argmax(bin_prob[i]).item() == 0:
                    pr_pos_neg = 'P'
                else:
                    pr_pos_neg = 'N'

                gt_hoi_inds = [ind for ind, v in enumerate(hoi_cates[i]) if v == 1]
                gt_hoi_int = hoi2int[gt_hoi_inds[0]]
                hoi_prob[i][:gt_hoi_int[0]] = 0
                hoi_prob[i][gt_hoi_int[1]+1:] = 0

                pr_hoi_cates = np.argsort(hoi_prob[i].cpu().data.numpy())[::-1][:1]
                pr_obj_cate_name = hoi_classes[gt_hoi_inds[0]].object_name()
                pr_vrb_cate_names = [hoi_classes[cate].verb_name() for cate in pr_hoi_cates]
                pr_hoi_cate_str = '%s - ' % pr_obj_cate_name + ','.join(pr_vrb_cate_names)

                print('GT: [%s] %s' % (gt_pos_neg, gt_hoi_cate_str))
                print('PR: [%s] %s' % (pr_pos_neg, pr_hoi_cate_str))
                # show_scores(hoi_classes, hoi_prob[i].cpu().data.numpy(), gt_hoi_int)

    error_bin_avg = error_bin_all / ins_count
    error_hoi_avg = error_hoi_all / ins_count
    print('[eval] avg bin error: %.4f   avg hoi error: %.4f' % (error_bin_avg, error_hoi_avg))
    return error_bin_avg, error_hoi_avg


if __name__ == '__main__':
    config_path = 'hico_spa.yaml'
    with open(config_path) as f:
        config = yaml.load(f)

    data_root = '../data/hico'
    data_save_dir = config['data_save_dir']
    hoi_db = prepare_hico(data_root, data_save_dir)
    test_dataset = HICODatasetSpa(hoi_db['val'])
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print('Loading models ...')
    model_save_dir = config['model_save_dir']
    model = SpaLan(config['lan_feature_dim'] + config['spa_feature_dim'],
            config['num_classes'])
    model = model.cuda()
    resume_dict = torch.load(os.path.join(model_save_dir, '%s_99_weights.pkl' % model))
    model.load_state_dict(resume_dict)
    model.eval()

    hoi_classes_path = os.path.join(data_root, 'hoi_categories.pkl')
    hoi_classes, obj_classes, vrb_classes, hoi2int = load_hoi_classes(hoi_classes_path)

    val(model, dataloader, hoi_classes, hoi2int, show=True)


