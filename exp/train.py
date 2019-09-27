import os
import time
import yaml

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import HICODatasetSpa
from load_data import prepare_hico, load_hoi_classes
from model import SpaLan
from val import val


def main(data_root, config):

    log_dir = 'logs'
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    logger = SummaryWriter(log_dir)

    print_freq = config['print_freq']
    save_freq = config['save_freq']
    data_save_dir = config['data_save_dir']
    model_save_dir = config['model_save_dir']
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    print('===== preparing =====')
    hoi_classes_path = os.path.join(data_root, 'hoi_categories.pkl')
    hoi_classes, _, _, hoi2int, obj2int = load_hoi_classes(hoi_classes_path)
    hoi_db = prepare_hico(data_root, data_save_dir)
    test_dataset = HICODatasetSpa(hoi_db['val'], obj2int)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_dataset = HICODatasetSpa(hoi_db['train'], obj2int)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    print('===== done =====')

    model = SpaLan(config['lan_feature_dim'] + config['spa_feature_dim'],
                   config['num_classes'])
    model = model.cuda()


    # Optimizer
    lr = config['learning_rate']
    lr_adjust_freq = config['lr_adjust_freq']
    wd = config['weight_decay']
    mt = config['momentum']

    batch_count = 0
    last_print_time = time.time()
    for epoch in range(config['n_epochs']):
        model.train()
        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=lr, momentum=mt, weight_decay=wd, nesterov=True)

        for data in train_dataloader:
            batch_count += 1

            spa_maps = Variable(data[0]).cuda()
            obj_vecs = Variable(data[1]).cuda()
            hoi_cates = Variable(data[2]).cuda()
            bin_cates = Variable(data[3]).cuda()
            int_mask = Variable(data[5]).cuda()

            pos_mask = torch.eq(bin_cates, 0)
            if pos_mask.sum().item() == 0:
                continue

            optimizer.zero_grad()
            bin_prob, hoi_prob, \
            loss_bin, loss_hoi, \
            error_bin, error_hoi = model(spa_maps, obj_vecs, hoi_cates, bin_cates, pos_mask, int_mask)

            loss = loss_bin + loss_hoi
            loss.backward()
            optimizer.step()

            logger.add_scalars('loss', {'all': loss.data.item(),
                                        'bin': loss_bin.data.item(),
                                        'hoi': loss_hoi.data.item()}, batch_count)
            logger.add_scalars('error', {'bin': error_bin.data.item(),
                                         'hoi': error_hoi.data.item()}, batch_count)

            if batch_count % print_freq == 0:
                curr_time = time.time()
                print('[Epoch %d][Batch %d] loss: %.4f time: %.2fs' % (epoch, batch_count, loss.data.item(),
                                                                       curr_time - last_print_time))
                print('\t\tloss_bin: %.4f\t\tloss_cls: %.4f'
                      % (loss_bin.data.item(), loss_hoi.data.item()))
                print('\t\terror_bin: %.4f\t\terror_hoi: %.4f'
                      % (error_bin.data.item(), error_hoi.data.item()))
                last_print_time = curr_time

        model.eval()
        error_bin_avg, error_hoi_avg = val(model, test_dataloader, hoi_classes, hoi2int, show=False)
        logger.add_scalars('error_val', {'bin': error_bin_avg,
                                         'hoi': error_hoi_avg}, epoch)
        if (epoch + 1) % save_freq == 0:
            model_file = os.path.join(model_save_dir, '%s_%d_weights.pkl' % (model, epoch))
            torch.save(model.state_dict(), model_file)
            np.save(os.path.join(model_save_dir, '%s_%d_lr.pkl' % (model, epoch)), lr)

        if (epoch + 1) % lr_adjust_freq == 0:
            lr = lr * 0.6

    logger.close()


if __name__ == '__main__':
    config_path = 'hico_spa.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f)

    data_root = '../data/hico'
    main(data_root, cfg)
