import os

import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

import CompositionalNets.Code.config as comp_net_config
from CompositionalNets.Initialization_Code.vMF_clustering import learn_vmf_clusters, save_cluster_images
from CompositionalNets.Initialization_Code.Learn_mix_model_vMF_view import learn_mix_model_vMF

from src.config import directories
from src.models import UNet
from src.dataset import ClassificationDataset

torch.manual_seed(42)

data_dir = directories['chaos']
unet_filename = 'unet_liver_2020-08-13_15:52:08.pth'
is_cuda_available = torch.cuda.is_available()
device = device = torch.device("cuda:0" if is_cuda_available else "cpu")

path_to_unet = os.path.join(directories['checkpoints'], unet_filename)
unet = UNet(pretrained=True)
unet.load_state_dict(torch.load(path_to_unet)['model_state_dict'])
if is_cuda_available: unet.to(device)
    
hdf5_path = os.path.join(directories['chaos'], 'train.hdf5')
with h5py.File(hdf5_path, 'r') as hf:
    images = hf['images'][:]

input_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])
to_grayscale = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

segmented_images = torch.empty(0)
for image in images:
    image = input_transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        out = unet(image)
        scores = F.softmax(out, dim=1)

        segmentations = torch.round(F.threshold(scores[:, 1, :, :], 0.9, 0))
        processed_image = segmentations.unsqueeze(1) * image
        processed_image = processed_image.detach().cpu()
        segmented_images = torch.cat((segmented_images, processed_image))

dataset = ClassificationDataset(segmented_images, [0] * len(segmented_images),
                               to_grayscale)
data_loader = DataLoader(dataset, batch_size=1)

model, loc_set =  learn_vmf_clusters(data_loader, img_per_cat=2000,
                                     u_out_name='chaos_pool5_512_u.pickle',
                                     p_out_name='chaos_pool5_512_p.pickle')

images = np.asarray([(to_grayscale(im).permute(1,2,0).numpy()*255).astype(np.uint8)
                     for im in segmented_images.detach().cpu()])

save_cluster_images(model, loc_set, in_images=images,
                    num_images=50, out_dir_name='cluster_images_chaos_pool5_512',
                    max_num_clusters=50, verbose=False)

import CompositionalNets.Code.config as comp_net_config
import CompositionalNets.Initialization_Code.config_initialization as comp_net_init_config
from CompositionalNets.Initialization_Code.comptSimMat import compute_similarity_matrix

compute_similarity_matrix(data_loader, 0, 'chaos_similarity_matrix', 
                          sim_dir_name='similarity_vgg_pool5_chaos',
                          u_out_name='chaos_pool5_512_u.pickle',
                          N_sub = 50)

learn_mix_model_vMF(data_loader, 0, sim_matrix_name='chaos_similarity_matrix',
                    sim_dir_name='similarity_vgg_pool5_chaos',
                    dict_filename='chaos_pool5_512_u.pickle',
                    mixdir_name=f'mix_model_vmf_chaos_EM_all/')
