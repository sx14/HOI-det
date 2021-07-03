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
from utils import eval, adjust_lr
os.environ['CUDA_VISIBLE_DEVICES']="3"

def main(config, target):
    print('========= training %s =========' % target)
    print('Loading data ...')
    # load labels
    with open(config['train_data'], 'rb') as f:
        train_data = pickle.load(f)
        train_feats = train_data['feature']
        if target == 'proposal':
            train_labels = train_data['bin_labels']
        else:
            train_labels = train_data['vrb_labels']
    with open(config['val_data'], 'rb') as f:
        val_data = pickle.load(f)
        val_feats = val_data['feature']
        if target == 'proposal':
            val_labels = val_data['bin_labels']
        else:
            val_labels = val_data['vrb_labels']
    # shuffle samples
    if config['shuffle']:
        index_shuf = list(range(len(train_labels)))
        np.random.shuffle(index_shuf)
        train_labels = train_labels[index_shuf]
        train_feats = train_feats[index_shuf]

    # create model
    print('Creating model ...')
    if target == 'proposal':
        model = FCLayers(config['feat_dim'], config['num_pro_cates'])
    else:
        model = FCLayers(config['feat_dim'], config['num_pre_cates'])
    if config['use_gpu']:
        model.cuda()



    eval(model, val_labels, val_feats, use_gpu=config['use_gpu'])
    # optimizer
    learning_rate = config['train_lr']
    momentum = config['train_momentum']
    weight_decay = config['train_weight_decay']
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)

    # use tensorboardX
    if config['use_log']:
        print('Using tensorboardX, run run_show.sh showing loss curve')
        if os.path.exists('logs'):
            shutil.rmtree('logs')
        logger = SummaryWriter('logs')

    # training
    batch_size = config['train_batch_size']
    train_sample_num = len(train_labels)
    train_batch_num = int(train_sample_num / batch_size)
    minibatch_range = list(range(train_batch_num))
    if target == 'proposal':
        loss_func = cross_entropy
    else:
        loss_func = binary_cross_entropy
    epoch = 0

    while epoch < config['train_epoch_num']:

        model.train()

        # shuffle batches
        if config['shuffle']:
            np.random.shuffle(minibatch_range)

        for minibatch_count, minibatch_index in enumerate(minibatch_range):

            optimizer.zero_grad()

            batch_feats = train_feats[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :]
            batch_feats = Variable(torch.from_numpy(batch_feats).type(torch.FloatTensor))
            batch_labels = train_labels[minibatch_index*batch_size:(minibatch_index+1)*batch_size, 0]
            batch_labels = Variable(torch.from_numpy(batch_labels).type(torch.LongTensor))

            if config['use_gpu']:
                batch_labels = batch_labels.cuda()
                batch_feats = batch_feats.cuda()

            prediction = model(batch_feats)
            loss_cls = loss_func(prediction, batch_labels)
            loss_cls.backward()
            optimizer.step()

            logger.add_scalars('train_'+target, {'loss': loss_cls.data.item()}, minibatch_count)

            if (minibatch_count+1) % config['train_print_freq'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      .format(epoch+1, (minibatch_count+1) * batch_size, train_sample_num,
                              100. * (minibatch_count+1) / train_batch_num, loss_cls.data.item()))

        # adjust learning rate
        epoch = epoch + 1
        learning_rate = adjust_lr(config['train_lr'],
                                  config['train_lr_adjust_rate'],
                                  config['train_lr_adjust_freq'],
                                  epoch)
        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=learning_rate, momentum=momentum,
                                    weight_decay=weight_decay, nesterov=True)
        print("Learning rate adjusted to %f" % learning_rate)

        # save weights
        if epoch % config['train_save_weight_freq'] == 0:
            if not os.path.exists(config['weights_dir']):
                os.makedirs(config['weights_dir'])
            model_file = os.path.join(config['weights_dir'], 'model_%s_%d.pkl' % (target, epoch))
            torch.save(model.state_dict(), model_file)
            eval(model, val_labels, val_feats)
            model.train()

    logger.close()
    print('========= training %s done =========\n\n' % target)


if __name__ == '__main__':

    with open('hoi_proposal/hoia_config.yaml', 'r') as f:
        config = yaml.load(f)

    main(config, 'proposal')
