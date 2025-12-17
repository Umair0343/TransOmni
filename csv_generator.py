"""
TransOmni - CSV Generator and File Renamer

This script:
1. Scans a data directory for images and masks
2. Optionally renames files to TransOmni-compatible format
3. Generates a CSV file for training/validation

Usage:
    python csv_generator.py --data_dir /path/to/data --output_csv data_list.csv

Current naming format detected:
    im_1.png, im_1_mask_tuft.png

TransOmni-compatible format:
    {name}.png, {name}_mask.png (with task_id derived from mask name)

The CSV will have columns:
    image_path, label_path, name, layer_id, task_id, scale_id
"""

import os
import os.path as osp
import glob
import argparse
import shutil
from collections import defaultdict

# Tissue type mapping
TISSUE_TYPES = {
    'tuft': 0,
    'cap': 1,
    'pt': 2,
    'dt': 3,
    'ptc': 4,
    'ves': 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate CSV and optionally rename files for TransOmni')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing images and masks')
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


def detect_tissue_type(mask_name):
    """Detect tissue type from mask filename."""
    mask_lower = mask_name.lower()
    
    # Check for each tissue type
    for tissue, task_id in TISSUE_TYPES.items():
        if tissue in mask_lower:
            # Special case: 'pt' should not match 'ptc'
            if tissue == 'pt' and 'ptc' in mask_lower:
                continue
            return task_id, tissue
    
    # Default to TUFT (task 0) if not detected
    return 0, 'tuft'


def find_image_mask_pairs(data_dir):
    """Find all image-mask pairs in directory."""
    pairs = []
    
    # Get all images (non-mask files)
    all_files = glob.glob(osp.join(data_dir, '*'))
    images = [f for f in all_files if osp.isfile(f) and 'mask' not in osp.basename(f).lower()]
    images = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    for image_path in sorted(images):
        base = osp.splitext(osp.basename(image_path))[0]
        ext = osp.splitext(image_path)[1]
        
        # Look for corresponding mask
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
            task_id, tissue = detect_tissue_type(osp.basename(mask_path))
            
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


def rename_files(pairs, data_dir, dry_run=False):
    """Rename files to TransOmni-compatible format.
    
    New format: {base_name}.png, {base_name}_mask.png
    """
    renamed_pairs = []
    
    for pair in pairs:
        old_image = pair['image_path']
        old_mask = pair['mask_path']
        base = pair['base_name']
        
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
        for i, pair in enumerate(pairs[:3]):
            print(f"  {pair['image_path']}, {pair['mask_path']}, task_id={pair['task_id']}")
        if len(pairs) > 3:
            print(f"  ... and {len(pairs) - 3} more entries")
        return
    
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
    random.shuffle(pairs)
    
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    
    return train_pairs, val_pairs


def main():
    args = parse_args()
    
    print(f"TransOmni CSV Generator")
    print(f"=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Rename files: {args.rename_files}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Find image-mask pairs
    print("Scanning for image-mask pairs...")
    pairs = find_image_mask_pairs(args.data_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    
    if not pairs:
        print("No image-mask pairs found!")
        return
    
    # Show tissue type distribution
    tissue_counts = defaultdict(int)
    for pair in pairs:
        tissue_counts[pair['tissue']] += 1
    
    print("\nTissue type distribution:")
    for tissue, count in sorted(tissue_counts.items()):
        task_id = TISSUE_TYPES[tissue]
        print(f"  {tissue.upper()} (task_id={task_id}): {count} images")
    
    # Rename files if requested
    if args.rename_files:
        print("\nRenaming files...")
        pairs = rename_files(pairs, args.data_dir, dry_run=args.dry_run)
    
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
