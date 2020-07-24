import pydicom

import PIL
from PIL import Image
import pdb
import numpy as np
from sklearn.preprocessing import normalize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os


class Chaos2DSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for the CHAOS CT/MR dataset
    Supports DICOM (.dcm) and PIL.Image supported images (.png, .jpg)
    """

    def __init__(self, image_pair_filepaths, input_image_handler=pydicom.dcmread, gt_image_handler=Image.open,
                 is_train=True, input_transform=None, gt_transform=None, cache=False, device=None):
        self.image_pair_filepaths = image_pair_filepaths
        self.is_train = is_train
        self.input_image_handler = input_image_handler
        self.gt_image_handler = gt_image_handler
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.cache = cache
        self.device = device
        self.cached_segmentation_pairs = []

        if cache:
            self._cache_segmentation_pairs()

    def _cache_segmentation_pairs(self):
        """Load all input image and ground truth images to the device memory"""
        self.cached_segmentation_pairs = []
        for input_image_fp, gt_image_fp in self.image_pair_filepaths:

            input_image, gt_image = self._load_image_pair(input_image_fp, gt_image_fp)
            self.cached_segmentation_pairs.append((input_image, gt_image))

    def _load_image_pair(self, input_image_fp, gt_image_fp):
        """Load the input image and ground truth images to the device memory"""
        input_image = self._get_array(self.input_image_handler(input_image_fp))
        gt_image = self._get_array(self.gt_image_handler(gt_image_fp))

#         if self.device:
#             input_image.to(self.device)
#             gt_image.to(self.device)

        return input_image, gt_image

    def _get_array(self, input_data):
        if type(input_data) == pydicom.dataset.FileDataset:
            image_arr = input_data.pixel_array
        elif type(input_data) == PIL.PngImagePlugin.PngImageFile:
            image_arr = np.array(input_data, dtype=np.uint8)
        else:
            raise NotImplementedError

        return image_arr

    def _transform(self, image, transform):
        if transform:
            image = transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image

    def __getitem__(self, key):
        if self.cache:
            input_image, gt_image = self.cached_segmentation_pairs[key]
        else:
            input_image_fp, gt_image_fp = self.image_pair_filepaths[key]
            input_image, gt_image = self._load_image_pair(input_image_fp, gt_image_fp)

        input_image = self._transform(input_image, self.input_transform)
        gt_image = self._transform(gt_image, self.gt_transform)

        return input_image, gt_image

    def __len__(self):
        return len(self.image_pair_filepaths)


class NormalizeInstance(object):
    """Normalize a numpy or tensor image with mean and standard deviation estimated
    from the sample itself.

    input_data: The array or tensor to normalize
    """
    
    def __call__(self, input_data):
        if type(input_data) == torch.Tensor:
            mean, std = input_data.mean(), input_data.std()
            input_data = F.normalize(input_data, [mean], [std])
        else:
            input_data = normalize(input_data)
        return input_data
    

def get_image_pair_filepaths(root_chaos_directory, modality='CT', is_train=True):
    """Returns a list of (image filepath, mask filepath) for every CHAOS dataset slice
    Args:
        root_chaos_directory: The root directory for the CHAOS dataset
            Expects the following file structure:
            root_chaos_directory/modality/patient/
                                         DICOM_anon/
                                             image1.ext1
                                             ...
                                             imageN.ext1
                                         Ground/
                                             image1.ext2
                                             ...
                                             imageN.ext2
    """
    assert modality == 'CT' or modality == 'MR', f'Modality can either be CT or MR, and not {modality}'
    if is_train:
        modality_path = os.path.join(root_chaos_directory, 'Train_Sets', modality)
    else:
        modality_path = os.path.join(root_chaos_directory, 'Test_Sets', modality)
    
    input_image_dir = 'DICOM_anon'
    gt_image_dir = 'Ground'
    image_pair_filepaths = []
    
    for patient in sorted(os.listdir(modality_path)):
        input_image_names = sorted(os.listdir(os.path.join(modality_path, patient, input_image_dir)))

        if is_train:
            
            gt_image_names = sorted(os.listdir(os.path.join(modality_path, patient, gt_image_dir)))
            assert len(input_image_names) == len(gt_image_names), f"Number of input images and segmentation masks don't match for patient {patient}"
            for input_name, gt_name in zip(input_image_names, gt_image_names):
                
                input_path = os.path.join(modality_path, patient, input_image_dir, input_name)
                gt_path = os.path.join(modality_path, patient, gt_image_dir, gt_name)
                image_pair_filepaths.append((input_path, gt_path))
        else:
            
            image_pair_filepaths.extend([os.path.join(modality_path, patient, input_image_dir, name) for name in input_image_names])
            
    return image_pair_filepaths
    