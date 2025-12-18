"""
TransOmni - CSV Generator and File Renamer

This script:
1. Scans a data directory for images and masks
2. Optionally renames files to TransOmni-compatible format
3. Generates a CSV file for training/validation

Usage:
    # Use hardcoded paths (for DT and PT data):
    python csv_generator.py --use_hardcoded --output_csv data_list.csv
    
    # Or specify a directory:
    python csv_generator.py --data_dir /path/to/data --output_csv data_list.csv

Naming formats supported:
    im_1.png, im_1_mask_tuft.png
    im_1.png, im_1_mask_distal.png (DT)
    im_1.png, im_1_mask_proximal.png (PT)

The CSV will have columns:
    image_path, label_path, name, layer_id, task_id, scale_id
"""

import os
import os.path as osp
import glob
import argparse
import shutil
from collections import defaultdict

# Tissue type mapping (Omni-Seg original)
TISSUE_TYPES_FULL = {
    'tuft': 0,
    'cap': 1,
    'pt': 2,
    'proximal': 2,  # Alias for PT
    'dt': 3,
    'distal': 3,    # Alias for DT
    'ptc': 4,
    'ves': 5,
}

# Simplified mapping for 2-class training (PT=0, DT=1)
# Use this when training only on PT and DT
TISSUE_TYPES_2CLASS = {
    'pt': 0,
    'proximal': 0,  # Alias for PT
    'dt': 1,
    'distal': 1,    # Alias for DT
}

# Hardcoded data paths for quick testing
HARDCODED_DATA_DIRS = [
    '/media/iml1/umair/unprocessed data/biopsy_dataset/pas-dt-data/pas-dt-data',
    '/media/iml1/umair/unprocessed data/biopsy_dataset/pas-pt-data',
]



def parse_args():
    parser = argparse.ArgumentParser(description='Generate CSV and optionally rename files for TransOmni')
    
    parser.add_argument('--data_dir', type=str, default='',
                        help='Directory containing images and masks')
    parser.add_argument('--use_hardcoded', action='store_true',
                        help='Use hardcoded data directories (DT and PT)')
    parser.add_argument('--output_csv', type=str, default='data_list.csv',
                        help='Output CSV file path')
    parser.add_argument('--rename_files', action='store_true',
                        help='Rename files to TransOmni-compatible format')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (for train/val split)')
    parser.add_argument('--split', action='store_true',
                        help='Create separate train/val CSV files')
    parser.add_argument('--scale_id', type=int, default=20,
                        help='Scale ID to use (default: 20)')
    parser.add_argument('--layer_id', type=int, default=0,
                        help='Layer ID to use (default: 0)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without making changes')
    
    return parser.parse_args()


def detect_tissue_type(mask_name, use_2class=False):
    """Detect tissue type from mask filename.
    
    Args:
        mask_name: Name of the mask file
        use_2class: If True, use PT=0, DT=1 mapping for 2-class training
    """
    mask_lower = mask_name.lower()
    
    # Detect tissue name
    if 'distal' in mask_lower:
        tissue = 'dt'
    elif 'proximal' in mask_lower:
        tissue = 'pt'
    elif 'ptc' in mask_lower:
        tissue = 'ptc'
    elif 'tuft' in mask_lower:
        tissue = 'tuft'
    elif 'cap' in mask_lower:
        tissue = 'cap'
    elif 'ves' in mask_lower:
        tissue = 'ves'
    elif 'dt' in mask_lower and 'ptc' not in mask_lower:
        tissue = 'dt'
    elif 'pt' in mask_lower and 'ptc' not in mask_lower:
        tissue = 'pt'
    else:
        tissue = 'tuft'  # Default
    
    # Get task_id based on mapping
    if use_2class:
        # 2-class mode: PT=0, DT=1
        task_id = TISSUE_TYPES_2CLASS.get(tissue, 0)
    else:
        # Full 6-class mode
        task_id = TISSUE_TYPES_FULL.get(tissue, 0)
    
    return task_id, tissue


def find_image_mask_pairs(data_dir, use_2class=False):
    """Find all image-mask pairs in directory.
    
    Args:
        data_dir: Directory to scan
        use_2class: If True, use PT=0, DT=1 mapping
    """
    pairs = []
    
    if not osp.exists(data_dir):
        print(f"Warning: Directory does not exist: {data_dir}")
        return pairs
    
    # Get all images (non-mask files)
    all_files = glob.glob(osp.join(data_dir, '*'))
    images = [f for f in all_files if osp.isfile(f) and 'mask' not in osp.basename(f).lower()]
    images = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    for image_path in sorted(images):
        base = osp.splitext(osp.basename(image_path))[0]
        ext = osp.splitext(image_path)[1]
        
        # Look for corresponding mask with various patterns
        mask_patterns = [
            osp.join(data_dir, f"{base}_mask*"),
            osp.join(data_dir, f"{base}*mask*"),
        ]
        
        mask_files = []
        for pattern in mask_patterns:
            mask_files.extend(glob.glob(pattern))
        
        # Filter to only mask files
        mask_files = [f for f in mask_files if 'mask' in osp.basename(f).lower()]
        
        if mask_files:
            mask_path = mask_files[0]
            task_id, tissue = detect_tissue_type(osp.basename(mask_path), use_2class)
            
            pairs.append({
                'image_path': image_path,
                'mask_path': mask_path,
                'base_name': base,
                'task_id': task_id,
                'tissue': tissue,
            })
        else:
            print(f"Warning: No mask found for {image_path}")
    
    return pairs


def find_image_mask_pairs_multi_dir(data_dirs, use_2class=False):
    """Find image-mask pairs from multiple directories."""
    all_pairs = []
    
    for data_dir in data_dirs:
        print(f"\nScanning: {data_dir}")
        pairs = find_image_mask_pairs(data_dir, use_2class)
        print(f"  Found {len(pairs)} image-mask pairs")
        all_pairs.extend(pairs)
    
    return all_pairs


def rename_files(pairs, data_dir, dry_run=False):
    """Rename files to TransOmni-compatible format.
    
    New format: {base_name}.png, {base_name}_mask.png
    """
    renamed_pairs = []
    
    for pair in pairs:
        old_image = pair['image_path']
        old_mask = pair['mask_path']
        base = pair['base_name']
        data_dir = osp.dirname(old_image)
        
        # Determine new names
        ext = osp.splitext(old_image)[1]
        new_image = osp.join(data_dir, f"{base}{ext}")
        new_mask = osp.join(data_dir, f"{base}_mask{ext}")
        
        # Check if already in correct format
        if old_image == new_image and old_mask == new_mask:
            renamed_pairs.append({
                **pair,
                'image_path': new_image,
                'mask_path': new_mask,
            })
            continue
        
        if dry_run:
            print(f"Would rename:")
            if old_image != new_image:
                print(f"  {osp.basename(old_image)} -> {osp.basename(new_image)}")
            if old_mask != new_mask:
                print(f"  {osp.basename(old_mask)} -> {osp.basename(new_mask)}")
        else:
            # Rename image if needed
            if old_image != new_image:
                if osp.exists(new_image):
                    print(f"Warning: {new_image} already exists, skipping")
                else:
                    shutil.move(old_image, new_image)
                    print(f"Renamed: {osp.basename(old_image)} -> {osp.basename(new_image)}")
            
            # Rename mask if needed
            if old_mask != new_mask:
                if osp.exists(new_mask):
                    print(f"Warning: {new_mask} already exists, skipping")
                else:
                    shutil.move(old_mask, new_mask)
                    print(f"Renamed: {osp.basename(old_mask)} -> {osp.basename(new_mask)}")
        
        renamed_pairs.append({
            **pair,
            'image_path': new_image,
            'mask_path': new_mask,
        })
    
    return renamed_pairs


def generate_csv(pairs, output_csv, scale_id=20, layer_id=0, dry_run=False):
    """Generate CSV file with image-mask pairs."""
    
    if dry_run:
        print(f"\nWould create CSV: {output_csv}")
        print("Sample entries:")
        for i, pair in enumerate(pairs[:5]):
            print(f"  {osp.basename(pair['image_path'])}, {osp.basename(pair['mask_path'])}, task_id={pair['task_id']} ({pair['tissue']})")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs) - 5} more entries")
        return
    
    # Create output directory if needed
    output_dir = osp.dirname(output_csv)
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_csv, 'w') as f:
        # Header
        f.write("image_path,label_path,name,layer_id,task_id,scale_id\n")
        
        # Data rows
        for pair in pairs:
            f.write(f"{pair['image_path']},{pair['mask_path']},{pair['base_name']},{layer_id},{pair['task_id']},{scale_id}\n")
    
    print(f"\nCreated CSV: {output_csv} with {len(pairs)} entries")


def split_train_val(pairs, train_ratio=0.8):
    """Split pairs into train and validation sets."""
    import random
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    
    n_train = int(len(pairs_copy) * train_ratio)
    train_pairs = pairs_copy[:n_train]
    val_pairs = pairs_copy[n_train:]
    
    return train_pairs, val_pairs


def main():
    args = parse_args()
    
    print(f"TransOmni CSV Generator")
    print(f"=" * 50)
    
    # Determine which directories to scan
    if args.use_hardcoded:
        print("Using hardcoded data directories:")
        for d in HARDCODED_DATA_DIRS:
            print(f"  - {d}")
        data_dirs = HARDCODED_DATA_DIRS
        use_2class = True  # Use PT=0, DT=1 mapping for 2-class training
        print("Using 2-class mapping: PT=0, DT=1")
    elif args.data_dir:
        print(f"Data directory: {args.data_dir}")
        data_dirs = [args.data_dir]
        use_2class = False  # Use full 6-class Omni-Seg mapping
    else:
        print("Error: Must specify --data_dir or --use_hardcoded")
        return
    
    print(f"Output CSV: {args.output_csv}")
    print(f"Rename files: {args.rename_files}")
    print(f"Dry run: {args.dry_run}")
    
    # Find image-mask pairs
    print("\nScanning for image-mask pairs...")
    if len(data_dirs) > 1:
        pairs = find_image_mask_pairs_multi_dir(data_dirs, use_2class)
    else:
        pairs = find_image_mask_pairs(data_dirs[0], use_2class)
    
    print(f"\nTotal found: {len(pairs)} image-mask pairs")
    
    if not pairs:
        print("No image-mask pairs found!")
        return
    
    # Show tissue type distribution
    tissue_counts = defaultdict(int)
    for pair in pairs:
        tissue_counts[pair['tissue']] += 1
    
    print("\nTissue type distribution:")
    for tissue, count in sorted(tissue_counts.items()):
        # Get the appropriate task_id for display
        task_ids = [p['task_id'] for p in pairs if p['tissue'] == tissue]
        task_id = task_ids[0] if task_ids else 0
        print(f"  {tissue.upper()} (task_id={task_id}): {count} images")
    
    # Rename files if requested
    if args.rename_files:
        print("\nRenaming files...")
        pairs = rename_files(pairs, data_dirs[0] if len(data_dirs) == 1 else '', dry_run=args.dry_run)
    
    # Split into train/val if requested
    if args.split:
        train_pairs, val_pairs = split_train_val(pairs, args.train_ratio)
        
        train_csv = args.output_csv.replace('.csv', '_train.csv')
        val_csv = args.output_csv.replace('.csv', '_val.csv')
        
        generate_csv(train_pairs, train_csv, args.scale_id, args.layer_id, args.dry_run)
        generate_csv(val_pairs, val_csv, args.scale_id, args.layer_id, args.dry_run)
        
        print(f"\nTrain set: {len(train_pairs)} images")
        print(f"Val set: {len(val_pairs)} images")
    else:
        generate_csv(pairs, args.output_csv, args.scale_id, args.layer_id, args.dry_run)
    
    print("\nDone!")
    
    if args.dry_run:
        print("\n(This was a dry run - no files were modified)")
        print("Remove --dry_run to apply changes")


if __name__ == '__main__':
    main()
