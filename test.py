"""
TransOmni - 2D Testing/Inference Script
Adapted from TransDoDNet's test.py for 2D histopathology image segmentation

Supports:
- Single image inference
- Batch inference on dataset
- Evaluation metrics (Dice, IoU)
"""

import argparse
import os
import sys

sys.path.append("..")

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import time
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.TransDoDNet import TransDoDNet as MOTS_model
from MOTSDataset import MOTSValDataSet, my_collate
from torch.utils.data import DataLoader


def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TransOmni Testing")

    # Data paths
    parser.add_argument("--data_dir", type=str, default='../data/val/',
                        help="Root directory of validation data")
    parser.add_argument("--val_list", type=str, default='',
                        help="Validation list file")
    parser.add_argument("--restore_from", type=str, default='snapshots/checkpoint.pth',
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default='predictions/',
                        help="Directory to save predictions")

    # Input configuration - 2D
    parser.add_argument("--input_size", type=str, default='256,256',
                        help="Input size (H,W)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Model settings
    parser.add_argument('--res_depth', default=50, type=int,
                        help="ResNet depth")
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8',
                        help="Dynamic head depth and width")
    parser.add_argument("--weight_std", type=str2bool, default=False,
                        help="Use weight standardization")

    # Transformer settings
    parser.add_argument("--using_transformer", type=str2bool, default=True)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=768, type=int)
    parser.add_argument('--hidden_dim', default=192, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=6, type=int)
    parser.add_argument('--num_feature_levels', default=3, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--normalize_before', default=False, type=str2bool)
    parser.add_argument('--deepnorm', default=True, type=str2bool)
    parser.add_argument("--add_memory", type=int, default=2)

    # GPU
    parser.add_argument("--gpu", type=str, default='0',
                        help="GPU device ID")
    parser.add_argument("--save_predictions", type=str2bool, default=True,
                        help="Save prediction images")

    return parser


def dice_score(pred, target):
    """Compute Dice score."""
    smooth = 1.0
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target):
    """Compute IoU score."""
    smooth = 1.0
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def main():
    """Main testing function."""
    parser = get_arguments()
    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Parse input size
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # Dynamic head configuration
    dep, wid = map(int, args.dyn_head_dep_wid.split(','))
    dyn_head_dep_wid = (dep, wid)

    # Create model
    model = MOTS_model(
        args,
        norm_cfg='IN',
        activation_cfg='relu',
        num_classes=args.num_classes,
        weight_std=args.weight_std,
        deep_supervision=False,
        res_depth=args.res_depth,
        dyn_head_dep_wid=dyn_head_dep_wid
    )

    # Load checkpoint
    if os.path.isfile(args.restore_from):
        print(f"Loading checkpoint from {args.restore_from}")
        checkpoint = torch.load(args.restore_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print(f"No checkpoint found at {args.restore_from}")
        return

    model.to(device)
    model.eval()

    # Create validation dataset
    val_dataset = MOTSValDataSet(
        args.data_dir,
        args.val_list,
        crop_size=input_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=my_collate
    )

    # Metrics storage
    all_dice = {i: [] for i in range(args.num_queries)}
    all_iou = {i: [] for i in range(args.num_queries)}

    print(f"Starting evaluation on {len(val_dataset)} images...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            images = torch.from_numpy(batch['image']).to(device).float()
            labels = torch.from_numpy(batch['label']).to(device).float()
            task_ids = torch.from_numpy(batch['task_id']).to(device)
            names = batch['name']

            # Forward pass
            preds = model(images, task_ids)

            # Process each sample in batch
            for i in range(images.shape[0]):
                task_id = int(task_ids[i])
                
                # Get prediction for this task
                pred = preds[i, task_id]  # [num_classes, H, W]
                pred = torch.sigmoid(pred)
                
                # Binary mask (class 0 for simplicity, adjust as needed)
                pred_binary = (pred[0] > 0.5).cpu().numpy().astype(np.float32)
                label = labels[i].cpu().numpy()

                # Compute metrics
                dice = dice_score(pred_binary, label)
                iou = iou_score(pred_binary, label)
                
                all_dice[task_id].append(dice)
                all_iou[task_id].append(iou)

                # Save prediction if enabled
                if args.save_predictions:
                    name = names[i] if isinstance(names[i], str) else str(batch_idx * args.batch_size + i)
                    save_path = os.path.join(args.output_dir, f'{name}_pred.png')
                    plt.imsave(save_path, pred_binary, cmap='gray')

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    tissue_names = ['TUFT', 'CAP', 'PT', 'DT', 'PTC', 'VES']
    
    total_dice = []
    total_iou = []
    
    for task_id in range(args.num_queries):
        if len(all_dice[task_id]) > 0:
            mean_dice = np.mean(all_dice[task_id])
            mean_iou = np.mean(all_iou[task_id])
            total_dice.extend(all_dice[task_id])
            total_iou.extend(all_iou[task_id])
            
            tissue_name = tissue_names[task_id] if task_id < len(tissue_names) else f'Task_{task_id}'
            print(f"{tissue_name:8s} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | "
                  f"Samples: {len(all_dice[task_id])}")

    print("-" * 50)
    print(f"{'Overall':8s} | Dice: {np.mean(total_dice):.4f} | IoU: {np.mean(total_iou):.4f} | "
          f"Samples: {len(total_dice)}")
    print("=" * 50)


if __name__ == '__main__':
    main()
