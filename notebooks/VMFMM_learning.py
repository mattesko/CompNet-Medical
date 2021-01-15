import os
import re
import gzip
import nibabel as nib
from PIL import Image
import cv2

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from CompositionalNets.Code.config import vc_num, backbone_type
from CompositionalNets.Initialization_Code.comptSimMat import compute_similarity_matrix
from CompositionalNets.Initialization_Code.vMF_clustering import learn_vmf_clusters, save_cluster_images
from CompositionalNets.Initialization_Code.Learn_mix_model_vMF_view import learn_mix_model_vMF
from CompositionalNets.Initialization_Code.config_initialization import nn_type

from src.config import Directories
from src.models import UNet
from src.dataset import ClassificationDataset, apply_ct_abdomen_filter
from src.utils import synthetic_occlusion

torch.manual_seed(42)

experiment = 35
num_mixture_models = 1
tight_crop = True
registration = False
slice_range = (-30, 30)

if registration:
    vol_dir = os.path.join(Directories.CHAOS_REGISTRATIONS, 'affine', 'out')
else:
    vol_dir = os.path.join(Directories.CHAOS_REGISTRATIONS, 'affine')
regex = re.compile('.*\.nii\.gz')
names = [f for f in sorted(os.listdir(vol_dir)) if regex.match(f)]
slices = []

for name in names:
    
    fp = os.path.join(vol_dir, name)
    with gzip.open(fp, 'rb') as f:

        niftii_object = nib.load(f.filename)
        volume = np.array(niftii_object.dataobj, dtype=np.int16)
        
        num_slices = volume.shape[2]
        amount = np.zeros(num_slices)

        for i in range(num_slices):

            s = volume[...,i]
            background = s.min()
            amount[i] = np.sum(s==background)
        
        idx = np.argmin(amount)
#         idx = 60

        curr_slices = volume[..., idx+slice_range[0]:idx+slice_range[1]]

        curr_slices = apply_ct_abdomen_filter(curr_slices)

        curr_min, curr_max = curr_slices.min(), curr_slices.max()
        curr_slices = (curr_slices - curr_min) / (curr_max - curr_min + 1e-12)

        curr_slices = np.transpose(curr_slices, (2, 0, 1))
        curr_slices = np.stack((curr_slices, curr_slices, curr_slices), axis=len(curr_slices.shape))
        curr_slices = np.rot90(curr_slices, k=1, axes=(1,2))
        curr_slices = curr_slices.astype(np.float32)
        
        # Crop around liver
        if tight_crop:
            curr_slices = [img[np.ix_((img[...,0]>0).any(1), (img[...,0]>0).any(0))] for img in curr_slices]

        slices.extend(curr_slices)

slices = np.asarray(slices)
train_slices, test_slices = train_test_split(slices, test_size=0.2, random_state=42)

dataset = ClassificationDataset(train_slices, [0] * len(train_slices), input_transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=1)

synthetic_images = synthetic_occlusion(test_slices[:], textured=False, color=0.78431374)
for i in range(8):
    try:
        im = Image.open(os.path.join(Directories.DATA, 'tumors', f'tumor_slice{i}.jpg'))
        im = (np.array(im).astype(np.float32) / 255)
        synthetic_images[i] = im
    except Exception as e:
        print(f'{e}')

vmf, loc_set =  learn_vmf_clusters(data_loader, img_per_cat=len(dataset), verbose=True,
                                     max_it=1000, tol=5e-12,
                                     u_out_name=f'chaos_pool5_{vc_num}_u_test_{experiment}.pickle',
                                     p_out_name=f'chaos_pool5_{vc_num}_p_test_{experiment}.pickle')

save_cluster_images(vmf, loc_set, in_images=train_slices*255,
                    num_images=16, out_dir_name=f'test_{experiment}',
                    max_num_clusters=20)

mat1, mat2 = compute_similarity_matrix(data_loader, 0, f'test_{experiment}',
                                       sim_dir_name=f'similarity_{nn_type}_pool5_chaos_{experiment}',
                                       u_out_name=f'chaos_pool5_{vc_num}_u_test_{experiment}.pickle',
                                       N_sub=min(200, len(dataset)//10), num_layer_features=min(100, len(dataset)))

learn_mix_model_vMF(data_loader, 0, sim_matrix_name=f'test_{experiment}',
                    num_layers=1, num_clusters_per_layer=num_mixture_models,
                    sim_dir_name=f'similarity_{nn_type}_pool5_chaos_{experiment}',
                    dict_filename=f'chaos_pool5_{vc_num}_u_test_{experiment}.pickle',
                    mixdir_name=f'mix_model_vmf_chaos_EM_all_test_{experiment}/',
                    im_channels=3)

from CompositionalNets.Code.config import categories, device_ids, categories_train, mix_model_path, dict_dir, layer, vMF_kappa, compnet_type, num_mixtures
from CompositionalNets.Code.config import config as cfg
from CompositionalNets.Code.model import Net
from CompositionalNets.Code.helpers import getVmfKernels, getCompositionModel
from CompositionalNets.Code.eval_occlusion_localization import visualize_response_map, eval_occ_detection
from CompositionalNets.Initialization_Code.config_initialization import extractor

occ_likely = [0.6 for _ in range(len(categories_train))]

dict_dir = os.path.join(Directories.COMPOSITIONAL_NETS,
                        f'models/init_{nn_type}/dictionary_{nn_type}/chaos_pool5_{vc_num}_u_test_{experiment}.pickle')
weights = getVmfKernels(dict_dir, device_ids)
mix_model_path = os.path.join(Directories.COMPOSITIONAL_NETS, 
                             f'models/init_{nn_type}/mix_model_vmf_chaos_EM_all_test_{experiment}/')
mix_models = getCompositionModel(device_ids, mix_model_path, layer,
                                 [0],
                                 compnet_type=compnet_type,
                                 num_mixtures=num_mixtures)

model = Net(extractor, weights, vMF_kappa, occ_likely, mix_models,
            bool_mixture_bg=True,
            compnet_type=compnet_type, num_mixtures=num_mixtures, 
            vc_thresholds=cfg.MODEL.VC_THRESHOLD, occlusion_threshold=19)
if device_ids:
    model.to(device_ids[0])
    
cc = transforms.Compose([
    transforms.ToTensor(),
])

i = 0
out_dir = f'{Directories.COMPOSITIONAL_NETS}/models/init_{encoder}/occlusion_maps/test_{experiment}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for image in synthetic_images[:]:
    
    if tight_crop:
        image = image[np.ix_((image[...,0]>0).any(1), (image[...,0]>0).any(0))]
    image = cc(image)

    if device_ids:
        image = image.cuda(device_ids[0])

    image = image.unsqueeze(0)
    try:
        with torch.no_grad():
            score, occ_maps, part_scores = model.get_occlusion(image, 0)

        occ_map = occ_maps[0].detach().cpu().numpy()
        occ_map = cv2.medianBlur(occ_map.astype(np.float32), 3)
        occ_img = visualize_response_map(occ_map, tit='', cbarmax=0)

        img_orig = (image[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        faco = img_orig.shape[0] / occ_img.shape[0]

        occ_img_s = cv2.resize(occ_img, (int(occ_img.shape[1] * faco), img_orig.shape[0]))[:,:img_orig.shape[1],:]

        canvas = np.concatenate((img_orig, occ_img_s), axis=1)
        cv2.imwrite(os.path.join(out_dir, f'{i:02d}.jpg'))
        i += 1
    except Exception as e:
        print(e)

        
data_dir = os.path.join(Directories.LITS, 'media', 'nas', '01_Datasets', 'CT', 'LITS')
data_train_dir = os.path.join(data_dir, 'Training Set')
data_test_dir = os.path.join(data_dir, 'Testing Set')

volume_filepaths = [os.path.join(data_train_dir, name) for name in sorted(os.listdir(data_train_dir)) 
                    if 'volume' in name]
segmentation_filepaths = [os.path.join(data_train_dir, name) for name in sorted(os.listdir(data_train_dir)) 
                          if 'segmentation' in name]

pairs = [(vol, gt) for vol, gt in zip(volume_filepaths, segmentation_filepaths)]

cc = transforms.Compose([
    transforms.ToTensor(),
])

out_dir = f'{Directories.COMPOSITIONAL_NETS}/models/init_{nn_type}/occlusion_maps/test_lits_{experiment}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for pair_num, (volume_fp, segmentation_fp) in enumerate(pairs[:]):
    i = 0
    
    volume = nib.load(volume_fp)
    segmentation = nib.load(segmentation_fp)
    
    volume_data = volume.get_fdata()
    segmentation_data = segmentation.get_fdata()
    vol_min = volume_data.min()
    vol_max = volume_data.max()
    
    _, _, num_slices = volume_data.shape
    
    test_images = []
    for j in range(num_slices):
        image = volume_data[...,j]
        target = segmentation_data[...,j]

        image = np.array(image)
        target = np.array(target)
        
        image = np.rot90(image, k=1)
        target = np.rot90(target, k=1)

        liver = target.copy()
        liver[target == 2] = 1
        
        tumor = target.copy()
        tumor[target == 1] = 0
        tumor[target == 2] = 1

        if np.sum(liver) > 10000:
            
            image = image * liver
            
            image = (image - image.min()) / (image.max() - image.min() + 1e-12)
            image = image.astype(np.float32)
            image = np.stack((image, image, image), axis=2)
            
            if tight_crop:
                image = image[np.ix_((image[...,0]>0).any(1), (image[...,0]>0).any(0))]
                
            image = cc(image)
            if device_ids:
                image = image.cuda(device_ids[0])
            image = image.unsqueeze(0)

            score, occ_maps, part_scores = model.get_occlusion(image, 0)
            occ_map = occ_maps[0].detach().cpu().numpy()
            occ_map = cv2.medianBlur(occ_map.astype(np.float32), 3)
            occ_img = visualize_response_map(occ_map, tit='', cbarmax=0)

            img_orig = (image[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            faco = img_orig.shape[0] / occ_img.shape[0]
            
            occ_img_s = cv2.resize(occ_img, (int(occ_img.shape[1] * faco), img_orig.shape[0]))[:,:img_orig.shape[1],:]
            canvas = np.concatenate((img_orig, occ_img_s, np.stack((target,target,target),axis=2)*255/2), axis=1)
            cv2.imwrite(os.path.join(out_dir, f'{pair_num:02d}_{i:02d}.jpg'), canvas)
            i += 1
