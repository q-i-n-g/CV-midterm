from __future__ import print_function, division
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
import numpy as np
import json

# Local modules
from utils.train import train_model, SGD_different_lr, alexnet_different_lr
from utils.transform import makeDefaultTransforms, makeNewTransforms, makeAggresiveTransforms
from utils.net_cub import init_resnet50, init_alexnet, init_resnet18, init_inceptionv3


def resnet50_cub_train(start='pretrain', num_epochs=80, batch_size=16, lr_new=1e-2, lr_fine=1e-3,
                       transform_method='new', device=None, dir="./models/classification", model_name=None):
    root_dir = 'CUB_200_2011'
    data_dir = os.path.join(root_dir, 'images_sorted')
    if model_name == None:
        if start == 'pretrain':
            model_name = 'resnet50_lrnew={}_lrfine={}_tranform:{}'.format(lr_new, lr_fine, transform_method)
        elif start == 'scratch':
            model_name = 'resnet50_from_stratch_lr={}_tranform:{}'.format(lr_new, transform_method)
    working_dir = os.path.join(dir, model_name)
    os.makedirs(working_dir, exist_ok=True)
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = 4
    if transform_method == 'new':
        data_transforms = makeNewTransforms()
    elif transform_method == 'default':
        data_transforms = makeDefaultTransforms()
    elif transform_method == 'aggressive':
        data_transforms = makeAggresiveTransforms()
    else:
        raise ValueError("you can only choose new or default or aggressive")

    # 导入数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    if start == 'pretrain':
        resnet_cub = init_resnet50()
        optimizer = SGD_different_lr(resnet_cub, lr_finetune=lr_fine, lr_new=lr_new, weight_decay=1e-4)

    elif start == 'scratch':
        resnet_cub = init_resnet50(pretrained=False)
        optimizer = optim.SGD(resnet_cub.parameters(), lr=lr_new, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError('you can only choose pretrain or scratch')

    resnet_cub.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # 训练
    model_ft = train_model(model=resnet_cub, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                           device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                           working_dir=working_dir, log_history=True)

    torch.save(model_ft.state_dict(), os.path.join(working_dir, 'model_weights.pth'))

    config = {
        "train": {
            "lr_new": lr_new,
            "lr_fine": lr_fine,
            "batch_size": batch_size,
            "transform_method": transform_method,
            "num_epoch": num_epochs,
            "dir": dir,
            "model_name": model_name,
            "train_set": start
        },
        "test": {
            "model_path": working_dir,
            "model_name": model_name,
            "transform_method": transform_method
        }
    }

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def resnet18_cub_train(start='pretrain', num_epochs=80, batch_size=16, lr_new=1e-2, lr_fine=1e-3,
                       transform_method='new', device=None, dir="./models/classification", model_name=None):
    root_dir = 'CUB_200_2011'
    data_dir = os.path.join(root_dir, 'images_sorted')
    if model_name == None:
        if start == 'pretrain':
            model_name = 'resnet18_lrnew={}_lrfine={}_tranform:{}'.format(lr_new, lr_fine, transform_method)
        elif start == 'scratch':
            model_name = 'resnet18_from_stratch_lr={}_tranform:{}'.format(lr_new, transform_method)
    working_dir = os.path.join(dir, model_name)
    os.makedirs(working_dir, exist_ok=True)
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = 4
    if transform_method == 'new':
        data_transforms = makeNewTransforms()
    elif transform_method == 'default':
        data_transforms = makeDefaultTransforms()
    elif transform_method == 'aggressive':
        data_transforms = makeAggresiveTransforms()
    else:
        raise ValueError("you can only choose new or default")

    # 导入数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    if start == 'pretrain':
        resnet_cub = init_resnet18()
        optimizer = SGD_different_lr(resnet_cub, lr_finetune=lr_fine, lr_new=lr_new, weight_decay=1e-4)

    elif start == 'scratch':
        resnet_cub = init_resnet18(pretrained=False)
        optimizer = optim.SGD(resnet_cub.parameters(), lr=lr_new, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError('you can only choose pretrain or scratch')

    resnet_cub.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # 训练
    model_ft = train_model(model=resnet_cub, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                           device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                           working_dir=working_dir, log_history=True)

    torch.save(model_ft.state_dict(), os.path.join(working_dir, 'model_weights.pth'))

    config = {
        "train": {
            "lr_new": lr_new,
            "lr_fine": lr_fine,
            "batch_size": batch_size,
            "transform_method": transform_method,
            "num_epoch": num_epochs,
            "dir": dir,
            "model_name": model_name,
            "train_set": start
        },
        "test": {
            "model_path": working_dir,
            "model_name": model_name,
            "transform_method": transform_method
        }
    }

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def alexnet_cub_train(start='pretrain', num_epochs=80, batch_size=16, lr_new=1e-2, lr_fine=1e-3, transform_method='new',
                      device=None, dir="./models/classification", model_name=None):
    root_dir = 'CUB_200_2011'
    data_dir = os.path.join(root_dir, 'images_sorted')
    if model_name == None:
        if start == 'pretrain':
            model_name = 'alexnet_lrnew={}_lrfine={}_tranform:{}'.format(lr_new, lr_fine, transform_method)
        elif start == 'scratch':
            model_name = 'alexnet_from_stratch_lr={}_tranform:{}'.format(lr_new, transform_method)
    working_dir = os.path.join(dir, model_name)
    os.makedirs(working_dir, exist_ok=True)
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = 4
    if transform_method == 'new':
        data_transforms = makeNewTransforms()
    elif transform_method == 'default':
        data_transforms = makeDefaultTransforms()
    elif transform_method == 'aggressive':
        data_transforms = makeAggresiveTransforms()
    else:
        raise ValueError("you can only choose new or default")

    # 导入数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    if start == 'pretrain':
        resnet_cub = init_alexnet()
        optimizer = alexnet_different_lr(resnet_cub, lr_finetune=lr_fine, lr_new=lr_new, weight_decay=1e-4)

    elif start == 'scratch':
        resnet_cub = init_alexnet(pretrained=False)
        optimizer = optim.SGD(resnet_cub.parameters(), lr=lr_new, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError('you can only choose pretrain or scratch')

    resnet_cub.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # 训练
    model_ft = train_model(model=resnet_cub, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                           device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                           working_dir=working_dir, log_history=True)

    torch.save(model_ft.state_dict(), os.path.join(working_dir, 'model_weights.pth'))

    config = {
        "train": {
            "lr_new": lr_new,
            "lr_fine": lr_fine,
            "batch_size": batch_size,
            "transform_method": transform_method,
            "num_epoch": num_epochs,
            "dir": dir,
            "model_name": model_name,
            "train_set": start
        },
        "test": {
            "model_path": working_dir,
            "model_name": model_name,
            "transform_method": transform_method
        }
    }

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


def inceptionv3_cub_train(start='pretrain', num_epochs=80, batch_size=16, lr_new=1e-2, lr_fine=1e-3,
                          transform_method='new', device=None, dir="./models/classification", model_name=None):
    root_dir = 'CUB_200_2011'
    data_dir = os.path.join(root_dir, 'images_sorted')
    if model_name == None:
        if start == 'pretrain':
            model_name = 'inceptionv3_lrnew={}_lrfine={}_tranform:{}'.format(lr_new, lr_fine, transform_method)
        elif start == 'scratch':
            model_name = 'inceptionv3_from_stratch_lr={}_tranform:{}'.format(lr_new, transform_method)
    working_dir = os.path.join(dir, model_name)
    os.makedirs(working_dir, exist_ok=True)
    if device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_workers = 4
    if transform_method == 'new':
        data_transforms = makeNewTransforms()
    elif transform_method == 'default':
        data_transforms = makeDefaultTransforms()
    elif transform_method == 'aggressive':
        data_transforms = makeAggresiveTransforms()
    else:
        raise ValueError("you can only choose new or default")

    # 导入数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    if start == 'pretrain':
        resnet_cub = init_inceptionv3()
        optimizer = SGD_different_lr(resnet_cub, lr_finetune=lr_fine, lr_new=lr_new, weight_decay=1e-4)

    elif start == 'scratch':
        resnet_cub = init_inceptionv3(pretrained=False)
        optimizer = optim.SGD(resnet_cub.parameters(), lr=lr_new, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError('you can only choose pretrain or scratch')

    resnet_cub.to(device)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # 训练
    model_ft = train_model(model=resnet_cub, criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                           device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                           working_dir=working_dir, log_history=True)

    torch.save(model_ft.state_dict(), os.path.join(working_dir, 'model_weights.pth'))

    config = {
        "train": {
            "lr_new": lr_new,
            "lr_fine": lr_fine,
            "batch_size": batch_size,
            "transform_method": transform_method,
            "num_epoch": num_epochs,
            "dir": dir,
            "model_name": model_name,
            "train_set": start
        },
        "test": {
            "model_path": working_dir,
            "model_name": model_name,
            "transform_method": transform_method
        }
    }

    with open(os.path.join(working_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

