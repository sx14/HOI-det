import os
import shutil
import pickle

import yaml
import numpy as np
import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from model import FCLayers
from utils import eval, adjust_lr, test
os.environ['CUDA_VISIBLE_DEVICES']="3"

def main(config, target):
    print('========= training %s =========' % target)
    print('Loading data ...')
    # load labels
    with open(config['test_data'], 'rb') as f:
        test_data = pickle.load(f)
        test_feats = test_data['feature']
        if target == 'proposal':
            test_labels = test_data['bin_labels']
        else:
            test_labels = test_data['vrb_labels']

    # create model
    print('Creating model ...')
    if target == 'proposal':
        model = FCLayers(config['feat_dim'], config['num_pro_cates'])
    else:
        model = FCLayers(config['feat_dim'], config['num_pre_cates'])
    model_weights = torch.load(config['test_weight_path'])
    model.load_state_dict(model_weights)
    if config['use_gpu']:
        model.cuda()

    test_pros = test(model, test_labels, test_feats, use_gpu=config['use_gpu'])
    test_data['results'] = test_pros
    with open(config['test_result_path'], 'wb') as f:
        pickle.dump(test_data, f)
        print('propsal results saved at: %s' % config['test_result_path'])


if __name__ == '__main__':

    with open('../hoi_proposal/hoia_config.yaml', 'r') as f:
        config = yaml.load(f)

    main(config, 'proposal')
