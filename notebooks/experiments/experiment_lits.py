import os
from PIL import Image
import pickle
import pdb

import h5py
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

from src.dataset import NormalizeInstance, ClassificationDataset
from src.utils import create_canvas
from src.models import UNet
from src.metrics import dice_loss, dice_score
from src.utils import create_canvas
from src.train import train_one_epoch, validate
from src.config import directories

data_dir = directories['chaos']
unet_filename = 'unet_liver_2020-08-13_15:52:08.pth'

from CompositionalNets.Code.config import categories, data_path, device_ids, categories_train, mix_model_path, dict_dir, layer, vMF_kappa, model_save_dir, compnet_type, backbone_type, num_mixtures
from CompositionalNets.Code.config import config as cfg
from CompositionalNets.Code.model import Net, resnet_feature_extractor
from CompositionalNets.Code.helpers import getVmfKernels, getCompositionModel, update_clutter_model
from CompositionalNets.Code.eval_occlusion_localization import visualize_response_map
from CompositionalNets.Code.losses import ClusterLoss

params = {
    "lr": 0.0001,
    "batch_size": 8,
    "split_train_val": 0.8,
    "epochs": 5,
    "use_dice_loss": False,
    "cache": True,
    "random_seed": 42,
    "shuffle_data": True,
    "scheduler": "ExponentialLR",
    "step_size": 15,
    "gamma": 0.75,
    "threshold": 0.9,
    'weight_decay': 4e-3
}

# categories_train = ['healty', 'malignant']
train_vmf_kernels = True
train_mixture_components = True
occ_likely = [0.6 for _ in range(len(categories_train))]

is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda_available else "cpu")
input_images_dtype = torch.double
targets_dtype = torch.long

# Load UNet for liver segmentation
path_to_unet = os.path.join(directories['checkpoints'], unet_filename)
unet = UNet(pretrained=True)
unet.load_state_dict(torch.load(path_to_unet)['model_state_dict'])
if is_cuda_available: unet.to(device)

input_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.Resize((128, 128)),
    transforms.CenterCrop((112, 112)),
#     transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
])

# Load data for training and validation
hdf5_path = os.path.join(directories['chaos'], 'train_augmented.hdf5')
hf = h5py.File(hdf5_path, 'r') 
image_dset = hf['images']

# CompNet takes whole image target labels
targets = [0] * len(image_dset)

# X_train, X_test, y_train, y_test = train_test_split(image_dset, targets,
#                                                   train_size=params['split_train_val'],
#                                                   random_state=params['random_seed'],
#                                                   shuffle=params["shuffle_data"])


X_train, y_train = image_dset, targets
train_dataset = ClassificationDataset(X_train, y_train, input_transform)
# val_dataset = ClassificationDataset(X_test, y_test, input_transform)

train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'])
# val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'])

num_train = len(train_dataset)
# num_val = len(val_dataset)

# Instantiate CompNet model, optimizer, and criterion
dict_dir = os.path.join(directories['CompositionalNets'], 
                        'models/init_vgg/dictionary_vgg/chaos_pool5_512_u.pickle')
weights = getVmfKernels(dict_dir, device_ids)
mix_model_path = os.path.join(directories['CompositionalNets'], 
                             'models/init_vgg/mix_model_vmf_chaos_EM_all/')
mix_models = getCompositionModel(device_ids, mix_model_path, layer, 
                                 [0],
                                 compnet_type=compnet_type,
                                 num_mixtures=num_mixtures)

extractor = models.vgg16(pretrained=True).features
extractor.cuda(device_ids[0]).eval()

model = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, 
            bool_mixture_bg=True,
            compnet_type=compnet_type, num_mixtures=num_mixtures, 
            vc_thresholds=cfg.MODEL.VC_THRESHOLD)

# pretrained_file = os.path.join(directories['CompositionalNets'], 'models', 'vgg_pool5_p3d+', 'best.pth')
# model.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

if not train_vmf_kernels:
    model.conv1o1.weight.requires_grad = False
else:
    model.conv1o1.weight.requires_grad = True

if not train_mixture_components:
    model.mix_model.requires_grad = False
else:
    model.mix_model.requires_grad = True

classification_loss = nn.CrossEntropyLoss()
cluster_loss = ClusterLoss()

optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), 
                                lr=params['lr'], weight_decay=params['weight_decay'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

# params['num_samples'] = num_train + num_val
print(f'Number of Training Images:\t{num_train}')
# print(f'Number of Validation Images:\t{num_val}')

best_check = {
    'epoch': 0,
    'best': 0,
    'val_acc': 0
}
alpha = 3  # vc-loss
beta = 3 # mix loss

# we observed that training the backbone does not make a very big difference and takes up a lot of memory
# if the backbone should be trained, then only with very small learning rate e.g. 1e-7
for param in model.backbone.parameters():
    param.requires_grad = True

print('Training')

num_accumulated_steps = 128 // params['epochs']

for epoch in range(params['epochs']):
    train_loss = 0.0
    correct = 0
    model.train()
    model.backbone.eval()
    for index, data in enumerate(train_dataloader):

        images, targets = data

        images = images.cuda(device_ids[0])
        targets = targets.cuda(device_ids[0])

        with torch.no_grad():
            out = unet(images)
            scores = F.softmax(out, dim=1)

            segmentations = torch.round(F.threshold(scores[:, 1, :, :], params['threshold'], 0))
            processed_images = segmentations.unsqueeze(1) * images

        output, vgg_feat, like = model(processed_images)

        # don't care about image classification
        out = output.argmax(1)
        correct += torch.sum(out == targets)
        class_loss = classification_loss(output, targets) / output.shape[0]
        loss = class_loss

        if alpha != 0:
            clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
            loss += alpha * clust_loss
        
#         pdb.set_trace()
        if beta!=0:
            mix_loss = like[0,targets[0]]
#             loss += -beta *mix_loss
            loss += beta * mix_loss

        #with torch.autograd.set_detect_anomaly(True):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # pseudo batches
#         if np.mod(index, num_accumulated_steps)==0:
#             optimizer.step()
#             optimizer.zero_grad()

        train_loss += loss.detach() * images.shape[0]

#     pdb.set_trace()
    updated_clutter = update_clutter_model(model, device_ids)
    model.clutter_model = updated_clutter
    scheduler.step()
    train_acc = correct.cpu().item() / num_train
    train_loss = train_loss.cpu().item() / num_train
#     out_str = f'Epochs: [{epoch}/{params["epochs"]}] Cluster Loss: {train_loss}'
    out_str = 'Epochs: [{}/{}], Train Acc:{}, Train Loss:{}'.format(epoch + 1, params['epochs'],
                                                                    train_acc, train_loss)
    print(out_str)
    
train_hdf5_fp = os.path.join(directories['lits'], 'train.hdf5')
with h5py.File(train_hdf5_fp, 'r') as hf:
    image_dataset = hf['images'][50:80]
    target_dataset = hf['masks'][50:80]

lits_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.Resize((512, 512)),
#     transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
])

test_dataset = ClassificationDataset(image_dataset, target_dataset, 
                                     input_transform=lits_transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=False)

idx = 30
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        image, mask = data
        
        if device_ids:
            image = image.cuda(device_ids[0])
            
        out = unet(image)
        scores = F.softmax(out, dim=1)

        segmentations = torch.round(F.threshold(scores[:, 1, :, :], 0.9, 0))
        processed_image = segmentations.unsqueeze(1) * image
        
        output, *_ = model(processed_image)
        
        #localize occluder
#         if i == 2: pdb.set_trace() # there's an indexing issue if not using a mixture background
        score, occ_maps, part_scores = model.get_occlusion(processed_image, 0)
        occ_map = occ_maps[0].detach().cpu().numpy()
        occ_map = cv2.medianBlur(occ_map.astype(np.float32), 3)
        occ_img = visualize_response_map(occ_map, tit='',cbarmax=0)
        
        # concatenate original image and occluder map
#         img_orig = np.array(Image.fromarray(image_dataset[i], mode='F').convert('RGB'))
        img_orig = (processed_image[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        mask = np.array(Image.fromarray(mask.squeeze().numpy()*255//2, mode='F').convert('RGB'))
        faco = img_orig.shape[0] / occ_img.shape[0]
        
        occ_img_s = cv2.resize(occ_img, (int(occ_img.shape[1] * faco), img_orig.shape[0]))
        mask = cv2.resize(mask, (int(occ_img.shape[1] * faco), img_orig.shape[0]))
        
        canvas = np.concatenate((img_orig, occ_img_s, mask), axis=1)
        plt.figure(figsize=(15, 15))
        plt.imshow(canvas)
        plt.axis('off')
        fp = f'{directories["CompositionalNets"]}/results/lits/augment/train_without_occluder/{i}.png'
        cv2.imwrite(fp, canvas)
#         print('Occlusion map written to: {}'.format(out_name))
        if i == idx: break
