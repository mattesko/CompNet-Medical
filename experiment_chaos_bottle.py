#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import os
import pdb
import warnings
import io

# from comet_ml import Experiment
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from dataset import Chaos2DSegmentationDataset, NormalizeInstance, get_image_pair_filepaths
from models import UNet
from metrics import dice_loss, dice_score, jaccard_score
from utils import create_canvas

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2 # 0: off, 2: on for all modules')
# os.chdir('CompositionalNets/')
# sys.path.append('/project/6052161/mattlk/workplace/CompNet')


# In[2]:
if __name__ == '__main__':

    # Change the below directory depending on where the CHAOS dataset is stored
    data_dir = os.path.join('CompositionalNets', 'data', 'chaos')


    # In[3]:


    # experiment = Experiment(api_key="P5seMqEJjqZ8mDA7QYSuK3yUJ",
    #                         project_name="chaos-liver-segmentation",
    #                         workspace="matthew42", auto_metric_logging=False)


    # # Train U-Net on CHAOS for Liver Segmentation

    # In[6]:


    # %%time
    params = {
        "lr": 0.0001,
        "batch_size": 16,
        "split_train_val": 0.8,
        "epochs": 1,
        "use_dice_loss": False,
        "random_seed": 42,
        "shuffle_data": True,
        "scheduler": "StepLR",
        "step_size": 15,
        "gamma": 0.75
    }

    lr = params['lr']
    batch_size = params['batch_size']
    split_train_val = params['split_train_val']
    epochs = params['epochs']
    use_dice_loss = params['use_dice_loss']
    random_seed = params['random_seed']
    shuffle_data = params["shuffle_data"]

    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda_available else "cpu")
    input_images_dtype = torch.double
    targets_dtype = torch.long
    cache_data = True

    cache_input_transform = transforms.Compose([
        NormalizeInstance(mean=255.0),
        transforms.Lambda(lambda x: x.astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ])

    cache_gt_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Resize((224, 224))
    ])

    input_transform = transforms.Compose([
    #     NormalizeInstance(mean=255.0),
    #     transforms.Lambda(lambda x: x.astype(np.uint8)),
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.RandomAffine(degrees=5, shear=5),
        transforms.ToTensor()
    ])

    gt_transform = transforms.Compose([
    #     transforms.Lambda(lambda x: x.astype(np.uint8)),
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.RandomAffine(degrees=5, shear=5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255),
        transforms.Lambda(lambda x: x.long()),
    ])

    # Load data for training and validation
    image_pair_filepaths = get_image_pair_filepaths(data_dir)[:20]
    train_filepaths, val_filepaths = train_test_split(image_pair_filepaths, train_size=split_train_val,
                                                      random_state=random_seed, shuffle=shuffle_data)
    # train_filepaths, val_filepaths = image_pair_filepaths, image_pair_filepaths

    train_dataset = Chaos2DSegmentationDataset(train_filepaths, input_transform=input_transform,
                                               gt_transform=gt_transform, cache=cache_data,
                                               cache_input_transform=cache_input_transform,
                                               cache_gt_transform=cache_gt_transform,
                                               device=device)

    val_dataset = Chaos2DSegmentationDataset(val_filepaths, input_transform=input_transform,
                                             gt_transform=gt_transform, cache=cache_data,
                                             cache_input_transform=cache_input_transform,
                                             cache_gt_transform=cache_gt_transform,
                                             device=device)

    num_train, num_val = len(train_dataset), len(val_dataset)
    params['num_samples'] = num_train + num_val
    print(f'Number of training images:\t{num_train}\nNumber of validation images:\t{num_val}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Instantiate model, optimizer, and criterion
    torch.cuda.empty_cache()
    unet = UNet(dice=use_dice_loss)
    # unet = UNet(in_channels=1, out_channels=1, padding=0)
    if is_cuda_available: unet = unet.to(device, dtype=input_images_dtype)

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    if params['scheduler'] == 'StepLR': 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.75)
    elif params['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # cross-entropy loss: weighting of negative vs positive pixels
    loss_weight = torch.DoubleTensor([0.01, 0.99])
    if is_cuda_available: loss_weight = loss_weight.to(device)
    criterion = dice_loss if use_dice_loss else CrossEntropyLoss(weight=loss_weight,
                                                                 reduction='mean')

    # experiment.log_parameters(params)


    # In[7]:


    # %%time
    num_accumulated_steps = 128 // batch_size

    # with experiment.train():
    print(f'Number of training images:\t{num_train}\nNumber of validation images:\t{num_val}')
    for epoch in tqdm(range(epochs), desc=f'Training {epochs} epochs'):

    #         pdb.set_trace()
        running_loss = 0.0
        unet.train()
        optimizer.zero_grad()
        scaled_loss = 0.0

        for i, data in enumerate(train_dataloader):

            input_images, targets = data

            if is_cuda_available:
                input_images = input_images.to(device, dtype=input_images_dtype)
                targets = targets.to(device, dtype=targets_dtype)

            outputs = unet(input_images)

    #             if use_dice_loss:
    #                 outputs = outputs[:,1,:,:].unsqueeze(dim=1)
    #                 loss = criterion(outputs, targets)
    #             else:
    #                 targets = targets.squeeze(dim=1)
    #                 loss = criterion(outputs, targets)

            targets = targets.squeeze(dim=1)
            loss = criterion(outputs, targets)

            loss.backward()
    #             scaled_loss += loss.item()
            running_loss += loss.detach()

            if i % num_accumulated_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    #                 running_loss += scaled_loss / num_accumulated_steps
                scaled_loss = 0.0

        if i % num_accumulated_steps != 0:
            optimizer.step()
    #             running_loss += scaled_loss / (i % num_accumulated_steps)
        running_loss = running_loss.item()
        if use_dice_loss:
            print(f'[Epoch {epoch+1:03d} Training]\tDice Loss:\t\t{running_loss/(i+1):.4f}')
        else:
            print(f'[Epoch {epoch+1:03d} Training]\tCross-Entropy Loss:\t{running_loss/(i+1):.4f}')
    #         experiment.log_metric("Running Loss", running_loss, epoch=epoch, step=epoch, include_context=False)

        unet.eval()
        all_f1 = torch.empty(0)
        all_jaccard = torch.empty(0)

        for i, data in enumerate(val_dataloader):
            accuracy = 0.0
            intersect = 0.0
            union = 0.0

            input_images, targets = data
            if is_cuda_available:
                input_images = input_images.to(device, dtype=input_images_dtype)
                targets = targets.to(device, dtype=targets_dtype)
            outputs = unet(input_images)

            # round outputs to either 0 or 1
    #             if not use_dice_loss: outputs = softmax(outputs)
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs[:, 1, :, :].unsqueeze(dim=1).round()
#             outputs, targets = outputs.data.cpu().numpy(), targets.data.cpu().numpy()
            outputs, targets = outputs.detach(), targets.detach()
            dice = dice_score(outputs, targets)
            jaccard = jaccard_score(outputs, targets)
            all_f1 = torch.stack((all_f1, dice)) if i == 0 else dice
            all_jaccard = torch.stack((all_jaccard, jaccard)) if i == 0 else jaccard

#             for out, gt in zip(outputs, targets):
#                 f1 = f1_score(targets.reshape(-1), outputs.reshape(-1), zero_division=1)
#                 all_f1.append(f1)
#                 jaccard = jaccard_score(targets.reshape(-1), outputs.reshape(-1))
#                 all_jaccard.append(jaccard)

            if i % 50 == 0 and epoch in [0, 25, 50, 75, 100, 125, 150]:
                outputs, targets = outputs.data.cpu().numpy(), targets.data.cpu().numpy()
                for idx, (out, gt) in enumerate(zip(outputs, targets)):
                    with warnings.catch_warnings():
                        img = create_canvas(out, gt, show=False)
    #                         warnings.filterwarnings("ignore",category=DeprecationWarning)
    #                         experiment.log_image(img, name=f'epoch_{epoch:03d}_batch_{i:03d}_idx_{idx}_segmap', overwrite=True, 
    #                                              image_format="png", image_scale=1.0, image_shape=None, image_colormap="gray",
    #                                              image_channels="first", copy_to_tmp=False, step=epoch)
    
        if params['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(all_f1.mean().item())
        else:
            scheduler.step()
        f1_mean, jaccard_mean = all_f1.mean().item(), all_jaccard.mean().item()
        print(f'[Epoch {epoch+1:03d} Validation]\tAverage F1 Score:\t{f1_mean:.4f}\tAverage Jaccard/IoU:\t{jaccard_mean:.4f}\n')

    #         experiment.log_metric('Validation Average F1 Score', np.mean(all_f1), 
    #                               epoch=epoch, include_context=False)
    #         experiment.log_metric('Validation Average Jaccard/IoU', np.mean(all_jaccard), 
    #                               epoch=epoch, include_context=False)

    # torch.save(unet.state_dict(), 'unet.pth')
    # experiment.log_asset('unet.pth', copy_to_tmp=False)
    # experiment.end()


    # In[ ]:




