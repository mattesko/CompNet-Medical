import random
import PIL
from PIL import Image
import pdb
import zipfile
import re
import os

import pydicom
import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py
import nibabel as nib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(self, X, y, input_transform=None, target_transform=None, 
                 seed=np.random.randint(2147483647)):
        self.X = X
        self.y = y
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.seed = seed
        
    def set_seed(self, seed):
        self.seed = seed
        
    def __getitem__(self, key):
        x, y = self.X[key], self.y[key]
        if self.input_transform and self.target_transform:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        if self.input_transform:
            x = self.input_transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.X)


class Chaos2DSegmentationDataset(Dataset):
    """
    PyTorch Dataset class for the CHAOS CT/MR dataset
    Supports DICOM (.dcm) and PIL.Image supported images (.png, .jpg)
    """

    def __init__(self, image_pair_filepaths, input_image_handler=pydicom.dcmread,
                 target_image_handler=Image.open, input_transform=None,
                 target_transform=None, cache=False):

        self.image_pair_filepaths = image_pair_filepaths
        self.input_image_handler = input_image_handler
        self.target_image_handler = target_image_handler
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.reset_seed = input_transform != None and target_transform != None
        self.cache = cache
        self.cached_segmentation_pairs = []
        self.seed = np.random.randint(2147483647)

        if cache:
            self._cache_segmentation_pairs()

    def _cache_segmentation_pairs(self):
        """Load all input image and ground truth images to the device memory"""
        for input_image_fp, target_image_fp in self.image_pair_filepaths:

            input_image, target_image = self._load_image_pair(input_image_fp, target_image_fp)
            self.cached_segmentation_pairs.append((input_image, target_image))

    def _load_image_pair(self, input_image_fp, target_image_fp):
        """Load the input image and ground truth images"""
        input_image = self._get_array(self.input_image_handler(input_image_fp))
        target_image = self._get_array(self.target_image_handler(target_image_fp))

        return input_image, target_image

    def _get_array(self, input_data):
        if type(input_data) == pydicom.dataset.FileDataset:
            image_arr = extract_array_as_HU(input_data)
            image_arr = apply_ct_abdomen_filter(image_arr)
        elif type(input_data) == PIL.PngImagePlugin.PngImageFile:
            image_arr = np.array(input_data, dtype=np.uint8)
        else:
            raise NotImplementedError

        return image_arr

    def __getitem__(self, key):
        if self.cache:
            input_image, target_image = self.cached_segmentation_pairs[key]
        else:
            input_image_fp, target_image_fp = self.image_pair_filepaths[key]
            input_image, target_image = self._load_image_pair(input_image_fp,
                                                          target_image_fp)
        
        # Need to use the same seed for the random package, so that any
        # random properties for both input and target transforms are the same
        if self.reset_seed:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        if self.input_transform:
            input_image = self.input_transform(input_image)
            
        if self.reset_seed:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return input_image, target_image

    def __len__(self):
        return len(self.image_pair_filepaths)
    
    
def extract_array_as_HU(dicom_obj):
    """Extract the pixel array and convert to Hounsfield Units"""
    array = dicom_obj.pixel_array
    array = array.astype(np.int16)
    
    array[array == -2000] = 0
    
    intercept = dicom_obj.RescaleIntercept
    slope = dicom_obj.RescaleSlope
    
    if slope != 1:
        array = slope * array.astype(np.float64)
        array = array.astype(np.int16)
        
    array += np.int16(intercept)
    return array


def apply_ct_abdomen_filter(array):
    """
    Apply a threshold (window of 350 and level of 40) to achieve a CT abdomen
    filtered array
    """
    L = 40
    W = 350
    array[array < (L-(W//2))] = L - (W//2)
    array[array > (L+(W//2))] = L + (W//2)
    return array


def get_CHAOS_abdomen_segmentation_pairs(data_dir):
    image_pair_filepaths = get_image_pair_filepaths(data_dir)
    X, y = [], []
    for dicom_fp, target_fp in image_pair_filepaths:
        
        dicom = pydicom.dcmread(dicom_fp)
        abdomen_image = extract_array_as_HU(dicom)
        abdomen_image = apply_ct_abdomen_filter(abdomen_image)
        X.append(abdomen_image)
        
        target_image = Image.open(target_fp)
        y.append(target_image)
    return X, y

    
class Hdf5SegmentationDataset(Dataset):
    def __init__(self, hdf5_path, image_dset_name, target_dset_name,
                 input_transform=None, target_transform=None, cache=False,
                 cache_input_transform=None, cache_target_transform=None,
                 distribution_name=None, target_count_name=None):

        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.image_dset_name = image_dset_name
        self.target_dset_name = target_dset_name 
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.cache = cache
        self.cache_input_transform = cache_input_transform
        self.cache_target_transform = cache_target_transform
        self.cached_segmentation_pairs = []
        self.target_distribution = self.hdf5_file[distribution_name][...] if distribution_name else None
        self.target_count = self.hdf5_file[target_count_name][...] if target_count_name else None
        self.seed = np.random.randint(2147483647)
        
        if self.cache:
            self._cache_segmentation_pairs()
            self.hdf5_file.close()
            
    def __getitem__(self, key):
        if self.cache:
            image, target = self.cached_segmentation_pairs[key]
        else:
            image = self.hdf5_file[self.image_dset_name][key]
            target = self.hdf5_file[self.target_dset_name][key]
            
        image = self._transform(image, self.input_transform)
        target = self._transform(target, self.target_transform)
        
        return image, target
    
    def __len__(self):
        if self.cache:
            return len(self.cached_segmentation_pairs)
        else:
            return len(self.hdf5_file[self.image_dset_name])
        
    def _cache_segmentation_pairs(self):
        image_dset = self.hdf5_file[self.image_dset_name][:]
        target_dset = self.hdf5_file[self.target_dset_name][:]
        for image, target in zip(image_dset, target_dset):

            image = self._transform(image, self.cache_input_transform)
            target = self._transform(target, self.cache_target_transform)
            
            self.cached_segmentation_pairs.append((image, target))
    
    def _transform(self, x, transform):
        if transform:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            return transform(x)
        else:
            return x
        

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


def get_chaos_volumes(root_chaos_directory, modality='CT', is_train=True):
    """Returns a list of slices grouped by volume/patien
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
    
    volumes = []
    for patient in sorted(os.listdir(modality_path)):
        
        input_image_names = sorted(os.listdir(os.path.join(modality_path, patient, input_image_dir)))
        pairs = []

        if is_train:
            
            gt_image_names = sorted(os.listdir(os.path.join(modality_path, patient, gt_image_dir)))
            assert len(input_image_names) == len(gt_image_names), f"Number of input images and segmentation masks don't match for patient {patient}"
            for input_name, gt_name in zip(input_image_names, gt_image_names):
                
                input_path = os.path.join(modality_path, patient, input_image_dir, input_name)
                gt_path = os.path.join(modality_path, patient, gt_image_dir, gt_name)
                pairs.append((input_path, gt_path))
        else:
            for name in input_image_names:
                input_path = os.path.join(modality_path, patient, input_image_dir, name)
                pairs.append((input_path, None))
        
        volumes.append(pairs)
            
    return volumes


class NormalizeInstance(object):
    """Normalize a numpy or tensor image with mean and standard deviation estimated
    from the sample itself.

    input_data: The array or tensor to normalize
    """
    
    def __init__(self, mean=None):
        self.mean = mean

    def __call__(self, input_data):
        if self.mean:
            return input_data * self.mean/input_data.max() - input_data.min()

        if type(input_data) == torch.Tensor:
            mean, std = input_data.mean(), input_data.std()
            input_data = F.normalize(input_data, [mean], [std])
        else:
            input_data = sk_normalize(input_data)
        return input_data
    

class Resize(object):
    """Resize a Tensor with the given size"""
    def __init__(self, size):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: F.interpolate(x, size=self.size)),
            transforms.Lambda(lambda x: x.squeeze(0))
        ])

    def __call__(self, image):
        return self.transform(image)
    
    
def filter_cxr_filepaths(cxr_filepaths, mask_filepaths):
    """
    Filter the Chest XRay filepaths to only contain ones where a respective 
    mask exists for it
    """
    p = re.compile('\d{4}')
    cxr_identifiers = [p.search(fp).group(0) for fp in cxr_filepaths]
    mask_identifiers = [p.search(fp).group(0) for fp in mask_filepaths]
    
    filtered_cxr_filepaths = []
    for i, cxr_idx in enumerate(cxr_identifiers):
        if cxr_idx in mask_identifiers:
            filtered_cxr_filepaths.append(cxr_filepaths[i])
    
    return filtered_cxr_filepaths
    
    
def create_cxr_hdf5(data_dir, train_size=0.8, out_dir_name='pulmonary_cxr_abnormalities'):
    """
    Create two hdf5 files containg the pulmonary Chest XRay images and masks
    """
    cxr_archive_fp = os.path.join(data_dir, 'pulmonary_cxr_abnormalities.zip')
    masks_archive_fp = os.path.join(data_dir, 'pulmonary_cxr_abnormalities_masks.zip')
    cxr_archive = zipfile.ZipFile(cxr_archive_fp)
    masks_archive = zipfile.ZipFile(masks_archive_fp)
    
    # Only want healthy lungs. Healthy instances are encoded as 0
    cxr_pattern = re.compile('ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/CHNCXR_\d*_0.png')
    cxr_filepaths = [fp for fp in sorted(cxr_archive.namelist()) if cxr_pattern.match(fp)]
    mask_pattern = re.compile('mask/CHNCXR_\d*_0_mask.png')
    mask_filepaths = [fp for fp in sorted(masks_archive.namelist()) if mask_pattern.match(fp)]

    cxr_filepaths = filter_cxr_filepaths(cxr_filepaths, mask_filepaths)

    assert len(mask_filepaths) == len(cxr_filepaths), 'Number of chext x ray and mask images don\'t match'
    cxr_arrays = []
    mask_arrays = []

    for cxr_fp, mask_fp in tqdm(zip(cxr_filepaths, mask_filepaths)):
        with cxr_archive.open(cxr_fp) as f_cxr, masks_archive.open(mask_fp) as f_mask:

            cxr = Image.open(f_cxr).resize((1024, 1024)).convert('RGB')
            cxr = np.array(cxr)
            cxr_arrays.append(cxr)

            mask = Image.open(f_mask).resize((1024, 1024)).convert('L')
            mask = np.array(mask)
            mask_arrays.append(mask)
            
    cxr_arrays = np.asarray(cxr_arrays, dtype=np.uint8)
    mask_arrays = np.asarray(mask_arrays, dtype=np.uint8)
    
    X_train, X_test, y_train, y_test = train_test_split(cxr_arrays, mask_arrays, 
                                                        train_size=train_size,
                                                       shuffle=False)
    
    hf_fp = os.path.join(data_dir, out_dir_name, 'train.hdf5')
    hf_train = h5py.File(hf_fp, 'w')
    
    hf_train.create_group('shenzhen')
    hf_train.create_group('shenzhen/healthy')
    hf_train.create_dataset('shenzhen/healthy/cxr', data=X_train, compression="lzf")
    hf_train.create_dataset('shenzhen/healthy/masks', data=y_train, compression="lzf")
    hf_train.close()
    
    hf_fp = os.path.join(data_dir, out_dir_name, 'test.hdf5')
    hf_test = h5py.File(hf_fp, 'w')
    
    hf_test.create_group('shenzhen')
    hf_test.create_group('shenzhen/healthy')
    hf_test.create_dataset('shenzhen/healthy/cxr', data=X_test, compression="lzf")
    hf_test.create_dataset('shenzhen/healthy/masks', data=y_test, compression="lzf")
    hf_test.close()
#     return hf_train, hf_test


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).
    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).
        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

        dreturn = {
            "input": input_slice,
            "gt": gt_slice
        }

        return dreturn


class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.
    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, cache=True,
                 input_transform=None, target_transform=None,
                 slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)

    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)

                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_input_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.input_transform = transform
        
    def set_target_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.
        :param transform: the new transformation
        """
        self.target_transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev)
        """
        sum_intensities = 0.0
        numel = 0

        with DatasetManager(self,
                            override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).
        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        if self.target_transform is not None:
            gt_img = self.input_transform(gt_img)

        return (input_img, gt_img)
