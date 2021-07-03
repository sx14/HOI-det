import numpy as np
import torch
import os
from torch.autograd import Variable
from torch.nn.functional import softmax

def adjust_lr(lr_init, lr_adjust_rate, lr_adjust_freq, curr_finished_epoch):
    # adjust learning rate AFTER N epochs
    lr_curr = lr_init * (lr_adjust_rate ** int(curr_finished_epoch / lr_adjust_freq))
    return lr_curr


def eval(model, labels, features, use_gpu=True, max_sample_num=6000):

    if use_gpu:
        model.cuda()

    model.eval()
    total = 0.0
    correct = 0.0

    sub_labels = labels[:max_sample_num]
    sub_features = features[:max_sample_num]

    batch_size = 20
    for index in range(0, len(sub_labels), batch_size):
        print('[test] %d/%d' % (index, len(sub_labels)))
        batch_labels = sub_labels[index: index+batch_size].astype('int32')
        batch_features = sub_features[index: index+batch_size, :].astype('float32')

        batch_features = Variable(torch.from_numpy(batch_features))
        batch_labels = Variable(torch.from_numpy(batch_labels).type(torch.LongTensor))

        if use_gpu:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()

        batch_scores = model(batch_features)
        batch_preds = batch_scores.data.max(1)[1]   # get the index of the max log-probability

        total += batch_labels.size(0)
        correct += batch_preds.eq(batch_labels.view(-1)).sum().item()
    print('Test Accuracy of the model: %f %%' % (100.0 * correct / total))
    return correct / total


def test(model, labels, features, use_gpu=True):

    if use_gpu:
        model.cuda()

    model.eval()

    sub_labels = labels
    sub_features = features
    sub_interactiveness = []

    batch_size = 20
    for index in range(0, len(sub_labels), batch_size):
        print('[test] %d/%d' % (index, len(sub_labels)))
        batch_labels = sub_labels[index: index+batch_size].astype('int32')
        batch_features = sub_features[index: index+batch_size, :].astype('float32')

        batch_features = Variable(torch.from_numpy(batch_features))
        batch_labels = Variable(torch.from_numpy(batch_labels).type(torch.LongTensor))

        if use_gpu:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()

        batch_scores = model(batch_features)
        batch_preds = batch_scores.data.max(1)[1]   # get the index of the max log-probability
        for pred in batch_preds:
            if pred == 0:
                sub_interactiveness.append(0.5)
            else:
                sub_interactiveness.append(1.0)
    return sub_interactiveness


def merge(vid_res_dir, res_path):
    wf = open(res_path, 'w')
    wf.write('{')
    wf.write('"version": "VERSION 1.0",')
    wf.write('"external_data": {"used": true, "detail": "Use pre-trained object detection model on ImageNet"},')
    wf.write('"results": {')

    vid_res_names = os.listdir(vid_res_dir)
    vid_res_num = len(vid_res_names)
    for i, vid_res_name in enumerate(vid_res_names):
        print('combine [%d/%d]' % (vid_res_num, i+1))
        vid_res_path = os.path.join(vid_res_dir, vid_res_name)
        with open(vid_res_path) as rf:
            json_content = json.load(rf)[vid_res_name.split('.')[0]]
        with open(vid_res_path) as rf:
            vid_res = rf.readline()

            if i == (vid_res_num-1):
                vid_res_str = vid_res[1:-1]
            else:
                vid_res_str = vid_res[1:-1]+','
            wf.write(vid_res_str)

    wf.write('}')
    wf.write('}')
    wf.close()
