"""
TransOmni - 2D Dataset Loader for Omni-Seg Histopathology Data
Adapted from TransDoDNet's MOTSDataset.py for 3D CT volumes

Key changes from TransDoDNet:
- 2D image loading (PNG) instead of 3D NIfTI volumes
- RGB (3 channels) instead of single-channel CT
- imgaug augmentations instead of batchgenerators
- Folder-based data organization (task_id_scale_id/stain_folder/)
- Edge weight calculation for loss weighting
- No HU truncation (histopathology images)
"""

import os
import os.path as osp
import numpy as np
import random
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import math
import glob
import scipy.ndimage
import cv2
from PIL import Image
import imgaug.augmenters as iaa
from skimage.transform import rescale, resize


class MOTSDataSet(data.Dataset):
    """Training dataset for Omni-Seg histopathology images.
    
    Supports 6 tissue types:
    - 0: TUFT (Tuft)
    - 1: CAP (Glomerular Capillary)
    - 2: PT (Proximal Tubule)
    - 3: DT (Distal Tubule)
    - 4: PTC (Peritubular Capillary)
    - 5: VES (Vessel)
    """
    
    def __init__(self, root, list_path='', max_iters=None, crop_size=(256, 256), 
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, 
                 edge_weight=1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        # Augmentation pipelines from Omni-Seg
        # Geometric augmentations (applied to both image and mask)
        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.ScaleX((0.75, 1.5)),
            iaa.ScaleY((0.75, 1.5))
        ])

        # Color augmentations (applied only to image)
        self.image_aug_color = iaa.Sequential([
            iaa.GammaContrast((0, 2.0)),
            iaa.Add((-0.1, 0.1), per_channel=0.5),
        ])

        # Noise augmentations (applied only to image)
        self.image_aug_noise = iaa.Sequential([
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
        ])

        # Resolution augmentation for low-res simulation
        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        # Parse folder structure to find all images
        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        
        # Find all task folders (format: {task_id}_{scale_id})
        tasks = glob.glob(os.path.join(self.root, '*'))
        
        for task_folder in tasks:
            if not os.path.isdir(task_folder):
                continue
            basename = os.path.basename(task_folder)
            parts = basename.split('_')
            if len(parts) < 2:
                continue
            try:
                task_id = int(parts[0])
                scale_id = int(parts[1])
            except ValueError:
                continue
                
            # Find stain folders within task folder
            stain_folders = glob.glob(os.path.join(task_folder, '*'))
            
            for stain_folder in stain_folders:
                if not os.path.isdir(stain_folder):
                    continue
                    
                # Find all images in stain folder
                images = glob.glob(os.path.join(stain_folder, '*'))
                
                for image_path in images:
                    if 'mask' in os.path.basename(image_path).lower():
                        continue
                    
                    # Find corresponding mask
                    _, ext = os.path.splitext(image_path)
                    mask_pattern = os.path.join(
                        stain_folder,
                        os.path.basename(image_path).replace(ext, '_mask*')
                    )
                    mask_files = glob.glob(mask_pattern)
                    
                    if mask_files:
                        task_list.append(task_id)
                        scale_list.append(scale_id)
                        image_path_list.append(image_path)
                        label_path_list.append(mask_files[0])

        self.files = []
        print("Start preprocessing....")
        
        for i in range(len(image_path_list)):
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            scale_id = scale_list[i]
            name = osp.basename(label_path)
            
            self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name,
                "task_id": task_id,
                "scale_id": scale_id,
            })
        
        # Repeat dataset if max_iters specified
        if max_iters is not None and len(self.files) > 0:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
            
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Read image and label
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])
        
        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]
        
        # Ensure 3 channels for image, handle different formats
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Ensure single channel for label
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        # Normalize image to 0-1 range if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize to crop size if different
        if image.shape[0] != self.crop_h or image.shape[1] != self.crop_w:
            image = resize(image, (self.crop_h, self.crop_w, 3), 
                          order=1, mode='constant', preserve_range=True)
            label = resize(label, (self.crop_h, self.crop_w), 
                          order=0, mode='constant', preserve_range=True)
        
        # Add batch dimension for imgaug
        image = np.expand_dims(image, axis=0)
        label_3ch = np.stack([label, label, label], axis=-1)
        label_3ch = np.expand_dims(label_3ch, axis=0)
        
        # Apply augmentations with probability
        seed = np.random.rand(4)
        
        if seed[0] > 0.5:
            image, label_3ch = self.image_mask_aug(images=image, heatmaps=label_3ch)
        
        if seed[1] > 0.5:
            image = self.image_aug_color(images=image)
        
        if seed[2] > 0.5:
            image = self.image_aug_noise(images=image)
        
        # Binarize label
        label = label_3ch[0, :, :, 0]
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0
        
        # Remove batch dimension
        image = image[0]
        
        # Transpose to Channel x H x W
        image = image.transpose((2, 0, 1))
        
        # Calculate edge weight if enabled
        if self.edge_weight:
            weight = scipy.ndimage.morphology.binary_dilation(
                label == 1, iterations=2
            ) & ~(label == 1).astype(bool)
            weight = weight.astype(np.float32)
        else:
            weight = np.ones(label.shape, dtype=np.float32)
        
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        return image.copy(), label.copy(), weight.copy(), name, task_id, scale_id


class MOTSValDataSet(data.Dataset):
    """Validation dataset for Omni-Seg histopathology images."""
    
    def __init__(self, root, list_path='', max_iters=None, crop_size=(256, 256),
                 mean=(128, 128, 128), scale=False, mirror=False, ignore_label=255,
                 edge_weight=1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        # Parse folder structure
        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        
        tasks = glob.glob(os.path.join(self.root, '*'))
        
        for task_folder in tasks:
            if not os.path.isdir(task_folder):
                continue
            basename = os.path.basename(task_folder)
            parts = basename.split('_')
            if len(parts) < 2:
                continue
            try:
                task_id = int(parts[0])
                scale_id = int(parts[1])
            except ValueError:
                continue
                
            stain_folders = glob.glob(os.path.join(task_folder, '*'))
            
            for stain_folder in stain_folders:
                if not os.path.isdir(stain_folder):
                    continue
                    
                images = glob.glob(os.path.join(stain_folder, '*'))
                
                for image_path in images:
                    if 'mask' in os.path.basename(image_path).lower():
                        continue
                    
                    _, ext = os.path.splitext(image_path)
                    mask_pattern = os.path.join(
                        stain_folder,
                        os.path.basename(image_path).replace(ext, '_mask*')
                    )
                    mask_files = glob.glob(mask_pattern)
                    
                    if mask_files:
                        task_list.append(task_id)
                        scale_list.append(scale_id)
                        image_path_list.append(image_path)
                        label_path_list.append(mask_files[0])

        self.files = []
        
        for i in range(len(image_path_list)):
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            scale_id = scale_list[i]
            name = label_path.replace('/', '-')
            
            self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name,
                "task_id": task_id,
                "scale_id": scale_id,
            })
            
        print('{} images are loaded!'.format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Read image and label
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])
        
        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]
        
        # Ensure 3 channels for image
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Ensure single channel for label
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        # Normalize image
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize if needed
        if image.shape[0] != self.crop_h or image.shape[1] != self.crop_w:
            image = resize(image, (self.crop_h, self.crop_w, 3),
                          order=1, mode='constant', preserve_range=True)
            label = resize(label, (self.crop_h, self.crop_w),
                          order=0, mode='constant', preserve_range=True)
        
        # Binarize label
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0
        
        # Transpose to Channel x H x W
        image = image.transpose((2, 0, 1))
        
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        weight = np.ones(label.shape, dtype=np.float32)
        
        return image.copy(), label.copy(), weight.copy(), name, task_id, scale_id


def my_collate(batch):
    """Custom collate function for DataLoader.
    
    Stacks batch items and returns a dictionary.
    """
    image, label, weight, name, task_id, scale_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    weight = np.stack(weight, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)
    scale_id = np.stack(scale_id, 0)
    
    data_dict = {
        'image': image,
        'label': label,
        'weight': weight,
        'name': name,
        'task_id': task_id,
        'scale_id': scale_id
    }
    
    return data_dict


if __name__ == '__main__':
    # Test the dataset loader
    from torch.utils.data import DataLoader
    
    trainset_dir = '/path/to/omniseg/train'
    batch_size = 2
    input_size = (256, 256)
    
    dataset = MOTSDataSet(
        trainset_dir, 
        list_path='',
        crop_size=input_size,
        scale=True,
        mirror=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=my_collate
    )
    
    for batch in loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Task IDs: {batch['task_id']}")
        break
