from net_train import resnet50_cub_train, alexnet_cub_train, resnet18_cub_train, inceptionv3_cub_train
from multiprocessing.dummy import freeze_support
from utils.data_sort import data_sort
from utils.set_seed import set_seed
import json
import os


def main():
    if not os.path.exists('./CUB_200_2011/images_sorted'):
        data_sort()
    set_seed(42)   
    with open('./models/final_model/config.json', 'r') as f:
        config = json.load(f)
    config = config['train']
    lr_new = config['lr_new']
    lr_fine = config['lr_fine']
    batch_size = config['batch_size']
    num_epochs = config['num_epoch']
    working_dir = config['dir']
    model_name = config['model_name']
    transform_method = config['transform_method']
    train_set = config['train_set']
    print(f"lr_fine={lr_fine},lr_new={lr_new},epoch={num_epochs}")

    inceptionv3_cub_train(train_set, num_epochs=num_epochs, lr_new=lr_new, lr_fine=lr_fine, batch_size=batch_size, transform_method=transform_method, model_name=model_name, dir=working_dir)

if __name__ == '__main__':
    freeze_support()
    main()
