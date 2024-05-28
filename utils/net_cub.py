from torchvision.models import resnet50, alexnet, resnet18, inception_v3
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import Inception_V3_Weights
import torch.nn as nn
from utils.weight_init import weight_init_kaiming


def init_resnet50(n_class=200, pretrained=True):
    if pretrained:
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    else:
        resnet = resnet50(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    return resnet


def init_alexnet(n_class=200, pretrained=True):
    if pretrained:
        alexnet_cub = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        alexnet_cub.classifier[6] = nn.Linear(alexnet_cub.classifier[6].in_features, n_class)
        alexnet_cub.classifier[6].apply(weight_init_kaiming)
    else:
        alexnet_cub = alexnet(weights=None)
        alexnet_cub.classifier[6] = nn.Linear(alexnet_cub.classifier[6].in_features, n_class)
        alexnet_cub.classifier[6].apply(weight_init_kaiming)
    return alexnet_cub

def init_resnet18(n_class=200, pretrained=True):
    if pretrained:
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    else:
        resnet = resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    return resnet

def init_inceptionv3(n_class=200, pretrained=True):
    if pretrained:
        resnet = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    else:
        resnet = inception_v3(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        resnet.fc.apply(weight_init_kaiming)
    return resnet

def init_net(model_name):
    if model_name.startswith('resnet50'):
        model = init_resnet50(pretrained=False)
    elif model_name.startswith('resnet18'):
        model = init_resnet18(pretrained=False)
    elif model_name.startswith('alexnet'):
        model = init_alexnet(pretrained=False)
    elif model_name.startswith('inceptionv3'):
        model = init_inceptionv3(pretrained=True)
    else:
        raise ValueError("wrong model name")
    return model

