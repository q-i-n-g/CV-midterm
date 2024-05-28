from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
from torchvision import datasets, models

#from imutils import paths
from pathlib import Path
import os, sys
import json

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.transform import makeDefaultTransforms, makeNewTransforms
from utils.load import unpickle, list_subdirectories
from utils.net_cub import init_net

def plot_loss_acc(model_name='resnet50', model_path=None):

    model = model_name
    root_dir = ''
    if model_path == None:
        model_root_dir = os.path.join(root_dir, 'models')

        output_dir = os.path.join(model_root_dir,'classification/{}'.format(model))
    else:
        output_dir = model_path
    model_history = os.path.join(output_dir,'model_history.pkl')

    history = unpickle(model_history)
    history['train_acc'] = [tensor.cpu().item() for tensor in history['train_acc']]
    history['test_acc'] = [tensor.cpu().item() for tensor in history['test_acc']]
    plt.figure(figsize=(18,10))

    plt.subplot(2,1,1)
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['train_loss'], 'b-', label='Train')
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['test_loss'], 'r-', label='Test')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss - Caltech Birds - {}'.format(model))
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['train_acc'], 'b-', label='Train')
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['test_acc'], 'r-', label='Test')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Accuracy - Caltech Birds - {}'.format(model))
    plt.legend()
    result_dir = 'result/figure/{}'.format(model)
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig('result/figure/{}/loss_acc.png'.format(model))


def every_class(model_name='resnet50', model_path=None, transform_method='new'):
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in stages}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in stages}
    class_names = image_datasets[stages[0]].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50_cub = init_net(model_name)
    resnet50_cub.load_state_dict(torch.load(model_file))
    resnet50_cub.to(device)
    resnet50_cub.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = resnet50_cub(inputs)
            _, preds = torch.max(outputs, 1)

            if i == 0:
                labels_truth = labels.cpu().numpy()
                labels_pred = preds.cpu().numpy()
                scores_pred = outputs.cpu().numpy()
            else:
                labels_truth = np.concatenate((labels_truth,labels.cpu().numpy()))
                labels_pred = np.concatenate((labels_pred,preds.cpu().numpy()))
                scores_pred= np.concatenate((scores_pred,outputs.cpu().numpy()))


    class_report_df = pd.DataFrame(classification_report(y_true=labels_truth, y_pred=labels_pred, target_names=class_names, output_dict=True))
    class_report_df.to_csv('result/csv/{}_classification.csv'.format(model), index=False)

    print('test acc:',sum(labels_truth == labels_pred)/len(labels_truth))
    result_dir = 'result/figure/{}'.format(model)
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(10,35))
    class_report_df.transpose()['precision'][:-3].sort_values().plot(kind='barh')
    plt.xlabel('Precision Score')
    plt.grid(True)
    plt.title(model)
    plt.savefig('result/figure/{}/Precision.png'.format(model))

    plt.figure(figsize=(10,35))
    class_report_df.transpose()['recall'][:-3].sort_values().plot(kind='barh')
    plt.xlabel('Recall Score')
    plt.title(model)
    plt.grid(True)
    plt.savefig('result/figure/{}/Recall.png'.format(model))

    plt.figure(figsize=(10,35))
    class_report_df.transpose()['f1-score'][:-3].sort_values().plot(kind='barh')
    plt.xlabel('F1 Score')
    plt.title(model)
    plt.grid(True)
    plt.savefig('result/figure/{}/F1.png'.format(model))


def plot_confusion_matrix(model_name='resnet50', model_path=None, transform_method='new'):
    model = model_name
    root_dir = ''
    stages = ['test']
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in stages}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=4)
                for x in stages}
    class_names = image_datasets[stages[0]].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50_cub = init_net(model_name)
    resnet50_cub.load_state_dict(torch.load(model_file))
    resnet50_cub.to(device)
    resnet50_cub.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)
            

            outputs = resnet50_cub(inputs)
            _, preds = torch.max(outputs, 1)

            if i == 0:
                labels_truth = labels.cpu().numpy()
                labels_pred = preds.cpu().numpy()
                scores_pred = outputs.cpu().numpy()
            else:
                labels_truth = np.concatenate((labels_truth,labels.cpu().numpy()))
                labels_pred = np.concatenate((labels_pred,preds.cpu().numpy()))
                scores_pred= np.concatenate((scores_pred,outputs.cpu().numpy()))

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true=labels_truth, y_pred=labels_pred), index=class_names, columns=class_names)
    plt.figure(figsize=(40,40))
    plt.imshow(confusion_matrix_df, cmap='Reds')
    plt.xticks(np.arange(0,len(class_names),1), class_names, rotation=90)
    plt.yticks(np.arange(0,len(class_names),1), class_names)
    plt.colorbar()
    plt.grid(True)
    plt.title('CalTech Birds 200 Dataset - Confusion Matrix - Model {}'.format(model))
    result_dir = 'result/figure/{}'.format(model)
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig('result/figure/{}/Confusion_Matrix'.format(model))


def all_plot(root):
    subdirs = list_subdirectories(root)
    plt.figure(figsize=(18, 10))
    for i in subdirs:
        output_dir = root + '/' + i + ''
        model_history = os.path.join(output_dir, 'model_history.pkl')

        history = unpickle(model_history)
        history['train_acc'] = [tensor.cpu().item() for tensor in history['train_acc']][:80]
        history['test_acc'] = [tensor.cpu().item() for tensor in history['test_acc']][:80]

        # plt.plot(np.arange(0, np.max(79) + 1, 1), history['train_acc'], label='Train')
        plt.plot(np.arange(0, np.max(79) + 1, 1), history['test_acc'], label=i)
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('top1 Acc')
        # plt.title('Training / Validation Accuracy - Caltech Birds - {}'.format(model))

        result_dir = 'result/figure/{}'.format('all')

    plt.legend()
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig('result/figure/{}/loss_acc_test.png'.format('all'))



# 使用示例
# root = './models/classification'
# subdirs = list_subdirectories(root)
# for i in subdirs:
#     plot_loss_acc(i)

# with open('/fdudata/conv_eval/eval/med_eval_conv/models/bluemodel/test/models/final_model/resnet50_CUB_from_strach/config.json','r') as f:
#     config = json.load(f)
# model_path = config['test']['model_path']
# model_name = config['test']['model_name']
# transform_method = config['test']['transform_method']

plot_confusion_matrix(model_name="inceptionv3", model_path="./models/final_model/inceptionv3", transform_method='new')
plot_loss_acc(model_name="inceptionv3", model_path="./models/final_model/inceptionv3")
every_class(model_name="inceptionv3", model_path="./models/final_model/inceptionv3", transform_method="new")

# root = './models/plot'
# all_plot(root)
