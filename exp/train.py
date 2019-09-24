import os
import time
import yaml

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import HICODatasetSpa
from load_data import prepare_hico
from model import SpaLan


def main(data_root, config):

    print_freq = config['print_freq']
    save_freq = config['save_freq']
    data_save_dir = config['data_save_dir']
    model_save_dir = config['model_save_dir']
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    print('===== preparing =====')
    hoi_db = prepare_hico(data_root, data_save_dir)
    test_dataset = HICODatasetSpa(hoi_db['val'])
    train_dataset = HICODatasetSpa(hoi_db['train'])
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print('===== done =====')

    model = SpaLan(config['lan_feature_dim'] + config['spa_feature_dim'],
                   config['num_classes'])
    model = model.cuda()
    model.train()

    # Optimizer
    lr = config['learning_rate']
    lr_adjust_freq = config['lr_adjust_freq']
    wd = config['weight_decay']
    mt = config['momentum']

    batch_count = 0
    last_print_time = time.time()
    for epoch in range(config['n_epochs']):

        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=lr, momentum=mt, weight_decay=wd, nesterov=True)

        for data in dataloader:
            batch_count += 1

            in_feats = Variable(data[0]).cuda()
            hoi_cates = Variable(data[1]).cuda()
            bin_cates = Variable(data[2]).cuda()

            pos_mask = torch.eq(bin_cates, 0)
            if pos_mask.sum().item() == 0:
                continue

            optimizer.zero_grad()
            bin_prob, hoi_prob, \
            loss_bin, loss_hoi, \
            error_bin, error_hoi = model(in_feats, hoi_cates, bin_cates, pos_mask)

            loss = loss_bin + loss_hoi
            loss.backward()
            optimizer.step()

            if batch_count % print_freq == 0:
                curr_time = time.time()
                print('[Epoch %d][Batch %d] loss: %.4f time: %.2fs' % (epoch, batch_count, loss.data.item(),
                                                                     curr_time - last_print_time))
                print('     loss_bin: %.4f      loss_cls: %.4f' % (loss_bin.data.item(), loss_hoi.data.item()))
                print('     error_bin: %d/%d       error_hoi: %d/%d' % (error_bin.data.item(), bin_cates.sum().data.item(),
                                                                        error_hoi.data.item(), hoi_cates[pos_mask].sum().data.item()))
                last_print_time = curr_time

        if (epoch + 1) % save_freq == 0:
            model_file = os.path.join(model_save_dir, '%s_%d_weights.pkl' % (model, epoch))
            torch.save(model.state_dict(), model_file)
            np.save(os.path.join(model_save_dir, '%s_%d_lr.pkl' % (model, epoch)), lr)

        if (epoch + 1) % lr_adjust_freq == 0:
            lr = lr * 0.1


if __name__ == '__main__':
    config_path = 'hico_spa.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f)

    data_root = '../data/hico'
    main(data_root, cfg)
