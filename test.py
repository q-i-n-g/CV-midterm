from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

from pathlib import Path
import os, sys
import json

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.transform import makeDefaultTransforms, makeNewTransforms, makeAggresiveTransforms
from utils.net_cub import init_resnet50, init_net
from utils.load import list_subdirectories

def test(model_name=None, model_path=None, transform_method='new'):
    model = model_name
    stages = ['test']
    root_dir = ''
    data_root_dir = os.path.join(root_dir, 'CUB_200_2011')
    data_dir = os.path.join(data_root_dir,'images_sorted')

    if model_path == None:
        model_root_dir = os.path.join(root_dir, 'models')
        output_dir = os.path.join(model_root_dir,'classification/{}'.format(model))
    else:
        output_dir = model_path
    model_file = os.path.join(output_dir, 'model_weights.pth')

    if transform_method == 'default':
        data_transforms = makeDefaultTransforms()
    elif transform_method == 'new':
        data_transforms = makeNewTransforms()
    elif transform_method == 'aggressive':
        data_transforms = makeAggresiveTransforms()
    else:
        raise ValueError("you can only choose new or default")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in stages}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in stages}
    class_names = image_datasets[stages[0]].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == None:
        net_cub = init_resnet50(pretrained=True)
    else:
        net_cub = init_net(model_name=model_name)
        net_cub.load_state_dict(torch.load(model_file))
    net_cub.to(device)
    net_cub.eval()

    top5_correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net_cub(inputs)
            if isinstance(outputs, tuple):
                # Model is Inception v3 with auxiliary outputs
                outputs, _ = outputs
            _, preds = torch.max(outputs, 1)
            _, top5_preds = torch.topk(outputs, 5, 1)

            correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
            top5_correct = top5_correct + correct.any(dim=1).sum().item()

            if i == 0:
                labels_truth = labels.cpu().numpy()
                labels_pred = preds.cpu().numpy()
                scores_pred = outputs.cpu().numpy()
            else:
                labels_truth = np.concatenate((labels_truth,labels.cpu().numpy()))
                labels_pred = np.concatenate((labels_pred,preds.cpu().numpy()))
                scores_pred= np.concatenate((scores_pred,outputs.cpu().numpy()))

    print('test acc:',sum(labels_pred == labels_truth)/len(labels_truth))
    print('top5 test acc:', top5_correct / len(labels_truth))
    # print(resnet50_cub)



# 使用示例
# root = './models/classification'
# subdirs = list_subdirectories(root)
# for i in subdirs:
#     print(i)
#     test(model_name=i, transform_method='new')
#     print('\n')

# test(model_name="resnet50_default", transform_method='default')

with open('./models/final_model/config.json','r') as f:
    config = json.load(f)
model_path = config['test']['model_path']
model_name = config['test']['model_name']
transform_method = config['test']['transform_method']

test(model_name=model_name, model_path=model_path, transform_method=transform_method)

