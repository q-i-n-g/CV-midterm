from __future__ import print_function, division

import os
import time
import copy
import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def save_pickle(pkl_object, fname):
    pkl_file = open(fname, 'wb')
    pickle.dump(pkl_object, pkl_file)
    pkl_file.close()


def SGD_different_lr(net, lr_finetune = 1e-3, lr_new = 1e-2, momentum=0.9, weight_decay=1e-4):
    finetune_params = []
    new_params = []
    for name, param in net.named_parameters():
        if name.startswith('fc'):  
            new_params.append(param)
        else:
            finetune_params.append(param)
    optimizer = optim.SGD([
        {'params': finetune_params, 'lr': lr_finetune}, 
        {'params': new_params, 'lr': lr_new} 
    ], momentum=0.9, weight_decay=weight_decay)

    return optimizer


def alexnet_different_lr(net, lr_finetune = 1e-3, lr_new = 1e-2, momentum=0.9, weight_decay=1e-4):
    optimizer = optim.SGD([
    {'params': net.classifier[6].parameters(), 'lr': lr_new},  # 对于最后一层使用较高的学习率
    {'params': net.features.parameters(), 'lr': lr_finetune}  # 对于前面的层使用较低的学习率
], momentum=momentum, weight_decay=weight_decay)
    return optimizer


def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, 
                num_epochs=25, return_history=False, log_history=True, working_dir='output'):
    since = time.time()
    writer = SummaryWriter(log_dir=os.path.join(working_dir, 'tensorboard_logs'))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'epoch' : [], 'train_loss' : [], 'test_loss' : [], 'train_acc' : [], 'test_acc' : []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    if isinstance(outputs, tuple):
                        # Model is Inception v3 with auxiliary outputs
                        outputs, aux_outputs = outputs
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2  # Combine losses
                    else:
                        loss = criterion(outputs, labels)
                        
                    _, preds = torch.max(outputs, 1)
            
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history['epoch'].append(epoch)
            history[phase+'_loss'].append(epoch_loss)
            history[phase+'_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # Log to TensorBoard
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        
        if log_history:
            save_pickle(history,os.path.join(working_dir,'model_history.pkl'))
        print()

    writer.close()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    print('Returning object of best model.')
    model.load_state_dict(best_model_wts)
    
    if return_history:
        return model, history
    else:
        return model