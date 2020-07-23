import pydicom

import PIL
from PIL import Image
import pdb
import numpy as np
from sklearn.preprocessing import normalize

import torch
from torch.utils.data import Dataset
import os

class SegmentationPair2D(object):
    def __init__(self, input_image_path, gt_image_path):
        
        self.input_image_path = input_image_path
        self.gt_image_path = gt_image_path

    def get_image_pair(self, input_image_handler, gt_image_handler):
        input_image = input_image_handler(self.input_image_path)
        gt_image = gt_image_handler(self.gt_image_path)
                
        return input_image, gt_image
        
    def get_input_image(self, input_image_handler):
        return input_image_handler(self.input_image_path)
    
    def get_gt_image(self, gt_image_handler):
        return gt_image_handler(self.gt_image_path)


class Chaos2DSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for the CHAOS CT/MR dataset
    
    Expects the following file structure:
        directory/modality/patient/
                                    input_image_dir/
                                        image1.ext1
                                        ...
                                        imageN.ext1
                                    gt_image_dir/
                                        image1.ext2
                                        ...
                                        imageN.ext2
    Supports DICOM (.dcm) and PIL.Image supported images (.png, .jpg)
    """
    def __init__(self, directory, modality='CT', input_image_dir='DICOM_anon', gt_image_dir='Ground',
                 input_image_handler=pydicom.dcmread, gt_image_handler=Image.open,
                 is_train=True, input_transform=None, gt_transform=None):
        assert modality == 'CT' or modality == 'MR', f'Modality can either be CT or MR, and not {modality}'
        
        self.directory = directory
        self.modality = modality
        self.is_train = is_train
        self.input_image_dir=input_image_dir
        self.gt_image_dir=gt_image_dir
        self.input_image_handler=input_image_handler
        self.gt_image_handler=gt_image_handler
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        
        self.segmentation_pairs = []
        
        if is_train:    
            self._load_image_pairs()
        else:
            self._load_input_images()
            
    def _load_input_images(self):
        modality_path = os.path.join(os.path.join(self.directory, self.modality))
        patients = sorted(os.listdir(modality_path))
        for patient in patients:
            
            image_names = sorted(os.listdir(modality_path, patient, 
                                            self.input_image_dir))
            
            input_image_filepaths = [os.path.join(modality_path, patient,
                                                  self.input_image_dir, 
                                                  name) for name in image_names]
            
            for input_image_fp in input_image_filepaths:
                self.segmentation_pairs.append(SegmentationPair2D(input_image_fp, None))

    def _load_image_pairs(self):
        modality_path = os.path.join(os.path.join(self.directory, self.modality))
        patients = sorted(os.listdir(modality_path))
        
        for patient in patients:
            
            input_image_names = sorted(os.listdir(os.path.join(modality_path,
                                                               patient, 
                                                               self.input_image_dir)))
            input_image_filepaths = [os.path.join(modality_path,
                                                  patient,
                                                  self.input_image_dir, 
                                                  name) for name in input_image_names]
            
            
            gt_image_names = sorted(os.listdir(os.path.join(modality_path,
                                                            patient,
                                                            self.gt_image_dir)))
            gt_image_filepaths = [os.path.join(modality_path,
                                               patient,
                                               self.gt_image_dir, 
                                               name) for name in gt_image_names]
            
            assert len(input_image_filepaths) == len(gt_image_filepaths), "Number of input images and segmentation masks don't match"

            for input_image_fp, gt_image_fp in zip(input_image_filepaths, gt_image_filepaths):
                self.segmentation_pairs.append(SegmentationPair2D(input_image_fp, gt_image_fp))

    def __getitem__(self, key):
        pair = self.segmentation_pairs[key]
        if self.is_train:
            input_data, gt_data = pair.get_image_pair(self.input_image_handler, self.gt_image_handler)
        else:
            input_data = pair.get_input_image(self.input_image_handler)
        
        if type(input_data) == pydicom.dataset.FileDataset:
            input_image_arr = input_data.pixel_array
            input_image_meta = input_data.file_meta
        elif type(input_data) == PIL.PngImagePlugin.PngImageFile:
            input_image_arr = np.array(input_data, dtype=np.uint8)
            input_image_meta = None
        else:
            raise NotImplementedError

        if type(gt_data) == pydicom.dataset.FileDataset:
            gt_image_arr = gt_data.pixel_array
            gt_image_arr = gt_data.file_meta
        elif type(gt_data) == PIL.PngImagePlugin.PngImageFile:
            gt_image_arr = np.array(gt_data, dtype=np.uint8)
            gt_image_meta = None
        elif not self.is_train:
            gt_image_arr = None
            gt_image_meta = None
        else:
            raise NotImplementedError
            
        if self.input_transform:
            input_image_arr = self.input_transform(input_image_arr)
        if self.gt_transform:
            gt_image_arr = self.gt_transform(gt_image_arr)

        data_dict = {
            'input': input_image_arr,
            'gt': gt_image_arr,
            'input_metadata': input_image_meta,
            'gt_metadata': gt_image_meta
        }
        return data_dict
        
    def __len__(self):
        return len(self.segmentation_pairs)
    

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