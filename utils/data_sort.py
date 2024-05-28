import os
import sys

from matplotlib import pyplot as plt
import random
import math
import shutil

def load_train_test_split(dataset_path=''):
    train_images = []
    test_images = []
    with open(os.path.join(dataset_path, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])
            if is_train:
                train_images.append(image_id)
            else:
                test_images.append(image_id)
    return train_images, test_images

def load_class_names(dataset_path=''):
    names = {}
    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])
    return names

def load_image_labels(dataset_path=''):
    labels = {}
    with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            labels[image_id] = class_id
    return labels
        
def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path
    return paths


def data_sort():
    root_data_dir = '../CUB_200_2011'
    old_images_dir = 'images'
    new_images_dir = 'images_sorted'

    train, test = load_train_test_split(dataset_path=root_data_dir)
    labels = load_image_labels(dataset_path=root_data_dir)
    image_paths = load_image_paths(dataset_path=root_data_dir)

    images_train_dir = os.path.join(root_data_dir,new_images_dir,'train')
    images_test_dir = os.path.join(root_data_dir,new_images_dir,'test')

    os.makedirs(os.path.join(root_data_dir,new_images_dir), exist_ok=True)
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)

    for image in train:
        new_dir = os.path.join(images_train_dir,image_paths[image].split('/')[0])
        old_path_image = os.path.join(root_data_dir,old_images_dir,image_paths[image])
        new_path_image = os.path.join(images_train_dir,image_paths[image])
        os.makedirs(new_dir, exist_ok=True)
        #os.symlink(old_path_image, new_path_image)
        shutil.move(old_path_image, new_path_image)

    for image in test:
        new_dir = os.path.join(images_test_dir,image_paths[image].split('/')[0])
        old_path_image = os.path.join(root_data_dir,old_images_dir,image_paths[image])
        new_path_image = os.path.join(images_test_dir,image_paths[image])
        os.makedirs(new_dir, exist_ok=True)
        #os.symlink(old_path_image, new_path_image)
        shutil.move(old_path_image, new_path_image)