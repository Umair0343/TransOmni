"""
TransOmni - 2D Dataset Loader for Omni-Seg Histopathology Data
Adapted from Omni-Seg's MOTSDataset_2D_Patch_supervise_csv.py

Supports:
- CSV-based data loading (like Omni-Seg)
- Folder-based data loading (TransOmni extension)
- Random cropping for large images (3000x3000 -> 512x512)
- imgaug augmentations matching Omni-Seg exactly
- Edge weight calculation for loss weighting
"""

import os
import os.path as osp
import numpy as np
import random
from torch.utils import data
import matplotlib.pyplot as plt
import math
import glob
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader, random_split
import scipy.ndimage
import sys


# Try to import pandas for CSV support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. CSV loading disabled.")


class MOTSDataSet(data.Dataset):
    """Training dataset for Omni-Seg histopathology images.
    
    Supports both CSV-based loading (Omni-Seg style) and folder-based loading.
    Handles large images (e.g., 3000x3000) with random cropping.
    
    Tissue types:
    - 0: TUFT (Glomerular Tuft)
    - 1: CAP (Glomerular Capillary)
    - 2: PT (Proximal Tubule)
    - 3: DT (Distal Tubule)
    - 4: PTC (Peritubular Capillary)
    - 5: VES (Vessel)
    """
    
    def __init__(self, root, list_path='', max_iters=None, crop_size=(512, 512), 
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, 
                 edge_weight=1):
        """Initialize dataset.
        
        Args:
            root: Path to CSV file OR root directory of images
            list_path: Unused (kept for compatibility)
            max_iters: Number of iterations per epoch (for repeating dataset)
            crop_size: Output crop size (H, W)
            mean: Mean values (unused, kept for compatibility)
            scale: Enable scaling augmentation
            mirror: Enable mirroring augmentation
            ignore_label: Label value to ignore
            edge_weight: Enable edge weighting in loss
        """
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        # Augmentation pipelines from Omni-Seg (exactly as original)
        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.ScaleX((0.75, 1.5)),
            iaa.ScaleY((0.75, 1.5))
        ])

        self.image_aug_color = iaa.Sequential([
            iaa.GammaContrast((0, 2.0)),
            iaa.Add((-0.1, 0.1), per_channel=0.5),
        ])

        self.image_aug_noise = iaa.Sequential([
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1)),
        ])

        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        self.image_aug_256 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((-10, 10), per_channel=0.5)
        ])

        # Cropping and padding for different input sizes
        self.crop512 = iaa.CropToFixedSize(width=512, height=512, position='uniform')
        self.pad512 = iaa.PadToFixedSize(width=512, height=512, position='uniform')
        
        # For cropping from 3000x3000 to crop_size
        self.crop_to_target = iaa.CropToFixedSize(
            width=self.crop_w, height=self.crop_h, position='uniform'
        )

        # Load data based on input type
        self.files = []
        self.use_csv = False
        
        if root.endswith('.csv') and HAS_PANDAS:
            # CSV-based loading (Omni-Seg style)
            self._load_from_csv(root)
        else:
            # Folder-based loading (TransOmni extension)
            self._load_from_folder(root)

        # Repeat dataset if max_iters specified
        if max_iters is not None and len(self.files) > 0:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

        self.now_len = len(self.files)
        print('{} images are loaded!'.format(self.now_len))

    def _load_from_csv(self, csv_path):
        """Load data from CSV file (Omni-Seg style)."""
        self.use_csv = True
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            self.files.append({
                "image": row["image_path"],
                "label": row["label_path"],
                "name": row.get("name", osp.basename(row["image_path"])),
                "layer_id": row.get("layer_id", 0),
                "task_id": row.get("task_id", 0),
                "scale_id": row.get("scale_id", 20),
            })

    def _load_from_folder(self, root_dir):
        """Load data from folder structure."""
        self.use_csv = False
        
        # Check if flat structure (like user's data) or nested structure
        images = glob.glob(os.path.join(root_dir, '*.png'))
        images += glob.glob(os.path.join(root_dir, '*.jpg'))
        images += glob.glob(os.path.join(root_dir, '*.tif'))
        
        # Filter out masks
        images = [f for f in images if 'mask' not in osp.basename(f).lower()]
        
        if images:
            # Flat folder structure
            for image_path in images:
                # Find corresponding mask
                base = osp.splitext(osp.basename(image_path))[0]
                mask_patterns = [
                    osp.join(root_dir, f"{base}_mask*.png"),
                    osp.join(root_dir, f"{base}_mask*.jpg"),
                    osp.join(root_dir, f"{base}*mask*.png"),
                ]
                
                mask_files = []
                for pattern in mask_patterns:
                    mask_files.extend(glob.glob(pattern))
                
                if mask_files:
                    # Determine task_id from mask name
                    mask_name = osp.basename(mask_files[0]).lower()
                    task_id = 0  # Default to TUFT
                    if 'tuft' in mask_name:
                        task_id = 0
                    elif 'cap' in mask_name:
                        task_id = 1
                    elif 'pt' in mask_name and 'ptc' not in mask_name:
                        task_id = 2
                    elif 'dt' in mask_name:
                        task_id = 3
                    elif 'ptc' in mask_name:
                        task_id = 4
                    elif 'ves' in mask_name:
                        task_id = 5
                    
                    self.files.append({
                        "image": image_path,
                        "label": mask_files[0],
                        "name": base,
                        "layer_id": 0,
                        "task_id": task_id,
                        "scale_id": 20,
                    })
        else:
            # Try nested folder structure (task_id_scale_id/stain/)
            tasks = glob.glob(os.path.join(root_dir, '*'))
            
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
                    
                    folder_images = glob.glob(os.path.join(stain_folder, '*'))
                    
                    for img_path in folder_images:
                        if 'mask' in osp.basename(img_path).lower():
                            continue
                        
                        _, ext = osp.splitext(img_path)
                        mask_pattern = osp.join(
                            stain_folder,
                            osp.basename(img_path).replace(ext, '_mask*')
                        )
                        mask_files = glob.glob(mask_pattern)
                        
                        if mask_files:
                            self.files.append({
                                "image": img_path,
                                "label": mask_files[0],
                                "name": osp.basename(img_path),
                                "layer_id": 0,
                                "task_id": task_id,
                                "scale_id": scale_id,
                            })

    def __len__(self):
        return self.now_len

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        # Read image and label
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])
        
        name = datafiles["name"]
        layer_id = datafiles["layer_id"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # Ensure 3 channels for image
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        else:
            image = image[:, :, :3]
        
        # Ensure proper format for label
        if len(label.shape) == 3:
            label = label[:, :, :3]
        else:
            label = np.stack([label, label, label], axis=-1)

        # Normalize to 0-1 range if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        if label.max() > 1.0:
            label = label.astype(np.float32) / 255.0

        # Add batch dimension for imgaug
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Handle different input sizes with cropping/padding
        img_h, img_w = image.shape[1], image.shape[2]
        
        if img_h >= 1024 or img_w >= 1024:
            # Large image (1024+ or 3000x3000): random crop to 512x512
            cnt = 0
            image_i, label_i = self.crop512(images=image, heatmaps=label)
            
            # Avoid patches that are mostly foreground (>80%)
            while label_i.sum() > 0.8 * 512 * 512 * 3 and cnt <= 50:
                image_i, label_i = self.crop512(images=image, heatmaps=label)
                cnt += 1
            
            image, label = image_i, label_i
            
        elif img_h == 256 or img_w == 256:
            # Small image: pad to 512x512
            image, label = self.pad512(images=image, heatmaps=label)

        # Apply augmentations with probability
        seed = np.random.rand(4)

        if seed[0] > 0.5:
            image, label = self.image_mask_aug(images=image, heatmaps=label)

        if seed[1] > 0.5:
            image = self.image_aug_color(images=image)

        if seed[2] > 0.5:
            image = self.image_aug_noise(images=image)

        # Binarize label
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0

        # Remove batch dimension and transpose to Channel x H x W
        image = image[0].transpose((2, 0, 1))
        label = label[0, :, :, 0]

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        # Calculate edge weight
        if self.edge_weight:
            weight = scipy.ndimage.morphology.binary_dilation(
                label == 1, iterations=2
            ) & ~label.astype(bool)
            weight = weight.astype(np.float32)
        else:
            weight = np.ones(label.shape, dtype=np.float32)

        label = label.astype(np.float32)

        return image.copy(), label.copy(), weight.copy(), name, layer_id, task_id, scale_id


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

        # Padding for consistent output size
        self.pad1024 = iaa.PadToFixedSize(width=1024, height=1024, position='center')
        self.pad512 = iaa.PadToFixedSize(width=512, height=512, position='center')

        # Load data
        self.files = []
        
        if root.endswith('.csv') and HAS_PANDAS:
            self.df = pd.read_csv(root)
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                self.files.append({
                    "image": row["image_path"],
                    "label": row["label_path"],
                    "name": row.get("name", osp.basename(row["image_path"])),
                    "layer_id": row.get("layer_id", 0),
                    "task_id": row.get("task_id", 0),
                    "scale_id": row.get("scale_id", 20),
                })
        else:
            # Folder-based loading
            images = glob.glob(os.path.join(root, '*.png'))
            images += glob.glob(os.path.join(root, '*.jpg'))
            images = [f for f in images if 'mask' not in osp.basename(f).lower()]
            
            for image_path in images:
                base = osp.splitext(osp.basename(image_path))[0]
                mask_files = glob.glob(osp.join(root, f"{base}_mask*.png"))
                mask_files += glob.glob(osp.join(root, f"{base}*mask*.png"))
                
                if mask_files:
                    mask_name = osp.basename(mask_files[0]).lower()
                    task_id = 0
                    if 'tuft' in mask_name:
                        task_id = 0
                    elif 'cap' in mask_name:
                        task_id = 1
                    elif 'pt' in mask_name and 'ptc' not in mask_name:
                        task_id = 2
                    elif 'dt' in mask_name:
                        task_id = 3
                    elif 'ptc' in mask_name:
                        task_id = 4
                    elif 'ves' in mask_name:
                        task_id = 5
                    
                    self.files.append({
                        "image": image_path,
                        "label": mask_files[0],
                        "name": base,
                        "layer_id": 0,
                        "task_id": task_id,
                        "scale_id": 20,
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
        layer_id = datafiles["layer_id"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        else:
            image = image[:, :, :3]
        
        if len(label.shape) == 3:
            label = label[:, :, :3]
        else:
            label = np.stack([label, label, label], axis=-1)

        # Normalize
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        if label.max() > 1.0:
            label = label.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Pad small images to 1024x1024 for validation
        img_h = image.shape[1]
        if img_h == 256 or img_h == 512:
            image, label = self.pad1024(images=image, heatmaps=label)

        # Binarize label
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0

        # Transpose
        image = image[0].transpose((2, 0, 1))
        label = label[0, :, :, 0]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        weight = np.ones(label.shape, dtype=np.float32)

        return image.copy(), label.copy(), weight.copy(), name, layer_id, task_id, scale_id


def my_collate(batch):
    """Custom collate function for DataLoader."""
    image, label, weight, name, layer_id, task_id, scale_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    weight = np.stack(weight, 0)
    name = np.stack(name, 0)
    layer_id = np.stack(layer_id, 0)
    task_id = np.stack(task_id, 0)
    scale_id = np.stack(scale_id, 0)
    
    data_dict = {
        'image': image,
        'label': label,
        'weight': weight,
        'name': name,
        'layer_id': layer_id,
        'task_id': task_id,
        'scale_id': scale_id
    }
    
    return data_dict


if __name__ == '__main__':
    # Test with folder-based loading
    trainset_dir = '/Users/muhammadumairshahid/Desktop/TUFT/pas-gtuft-data/pas-gtuft-data'
    batch_size = 4
    input_size = (512, 512)

    print("Testing folder-based loading with 3000x3000 images...")
    
    trainloader = DataLoader(
        MOTSDataSet(
            trainset_dir, 
            list_path='',
            max_iters=250 * batch_size,
            crop_size=input_size,
            scale=True,
            mirror=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=my_collate
    )

    for iter_idx, batch in enumerate(trainloader):
        print(f"Batch {iter_idx}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        print(f"  Task IDs: {batch['task_id']}")
        if iter_idx >= 2:
            break
    
    print("\nDataset test completed!")
