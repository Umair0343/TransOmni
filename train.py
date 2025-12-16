"""
TransOmni - 2D Training Script
Adapted from TransDoDNet's train.py for 2D histopathology image segmentation

Key changes from TransDoDNet:
- 2D input size (H, W) instead of 3D (D, H, W)
- 3-channel input (RGB) instead of 1-channel (CT)
- 6 tissue types (Omni-Seg) instead of 7 organs (MOTS)
- Edge weight support in loss computation
- imgaug-based augmentations in dataset
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
import random
import timeit
import logging
from math import ceil
from tqdm import tqdm

from models.TransDoDNet import TransDoDNet as MOTS_model
from MOTSDataset import MOTSDataSet, my_collate
from loss_functions import loss
from utils.ParaFlop import print_model_parm_nums
import utils.my_utils as utils
from engine import Engine

# Try to import apex for mixed precision training
try:
    from apex import amp
    HAS_APEX = True
except ImportError:
    HAS_APEX = False
    print("APEX not found, mixed precision training disabled")

start = timeit.default_timer()


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
    parser = argparse.ArgumentParser(description="TransOmni - 2D Multi-Tissue Segmentation")

    # Data paths
    parser.add_argument("--data_dir", type=str, default='../data/',
                        help="Root directory of training data")
    parser.add_argument("--train_list", type=str, default='',
                        help="Training list file (optional, uses folder structure)")
    parser.add_argument("--val_list", type=str, default='',
                        help="Validation list file")
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/',
                        help="Directory to save checkpoints")
    parser.add_argument("--reload_path", type=str, default='snapshots/checkpoint.pth',
                        help="Path to checkpoint for resuming")
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False,
                        help="Resume training from checkpoint")

    # Input configuration - 2D
    parser.add_argument("--input_size", type=str, default='256,256',
                        help="Input size (H,W)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument('--local_rank', type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--FP16", type=str2bool, default=False,
                        help="Use mixed precision training (requires APEX)")
    parser.add_argument("--num_epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--itrs_each_epoch", type=int, default=250,
                        help="Iterations per epoch")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Starting epoch")
    parser.add_argument("--val_pred_every", type=int, default=50,
                        help="Validation frequency (epochs)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--lr_linear_proj_mult", type=float, default=0.1,
                        help="LR multiplier for linear projections")
    parser.add_argument("--lr_tr_mult", type=float, default=1.0,
                        help="LR multiplier for transformer")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Optimizer settings
    parser.add_argument("--weight_std", type=str2bool, default=False,
                        help="Use weight standardization")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Learning rate decay power")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")

    # Data augmentation
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="Ignore label value")
    parser.add_argument("--random_mirror", type=str2bool, default=True,
                        help="Use random mirroring")
    parser.add_argument("--random_scale", type=str2bool, default=True,
                        help="Use random scaling")
    parser.add_argument("--random_seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--edge_weight", type=int, default=1,
                        help="Use edge weighting in loss")

    # Others
    parser.add_argument("--gpu", type=str, default='None',
                        help="GPU device ID")

    # ResNet settings
    parser.add_argument('--res_depth', default=50, type=int,
                        help="ResNet depth (18, 34, 50, 101)")
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8',
                        help="Dynamic head depth and width")

    # Transformer settings
    parser.add_argument("--using_transformer", type=str2bool, default=True,
                        help="Use deformable transformer")
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'), help="Position embedding type")
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoder layers")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoder layers")
    parser.add_argument('--dim_feedforward', default=768, type=int,
                        help="Feedforward dimension")
    parser.add_argument('--hidden_dim', default=192, type=int,
                        help="Hidden dimension")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout rate")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads")
    parser.add_argument('--num_queries', default=6, type=int,
                        help="Number of queries (one per tissue type)")
    parser.add_argument('--pre_norm', action='store_true',
                        help="Use pre-normalization")
    parser.add_argument('--num_feature_levels', default=3, type=int,
                        help="Number of feature levels")
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help="Decoder sampling points")
    parser.add_argument('--enc_n_points', default=4, type=int,
                        help="Encoder sampling points")
    parser.add_argument('--normalize_before', default=False, type=str2bool,
                        help="Normalize before attention")
    parser.add_argument('--deepnorm', default=True, type=str2bool,
                        help="Use DeepNorm initialization")
    parser.add_argument("--add_memory", type=int, default=2, choices=(0, 1, 2),
                        help="Feature fusion: 0=cnn, 1=transformer, 2=cnn+transformer")

    # Optimizer type
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=('sgd', 'adamw'), help="Optimizer type")

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    """Polynomial learning rate decay."""
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, args):
    """Adjust learning rate with warmup and decay."""
    lr = args.learning_rate
    num_steps = args.num_epochs
    power = args.power
    
    # Warmup for first 10 epochs
    if i_iter < 10:
        lr = 1e-2 * lr + i_iter * (lr - 1e-2 * lr) / 10.
    else:
        lr = lr_poly(lr, i_iter, num_steps, power)
    
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * args.lr_tr_mult
    
    return lr


def dice_score(preds, labels):
    """Compute Dice score on GPU."""
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den
    return dice.mean()


def compute_dice_score(preds, labels):
    """Compute Dice score for predictions."""
    preds = torch.sigmoid(preds)
    pred_pa = preds[:, 0, :, :]
    label_pa = labels[:, 0, :, :] if len(labels.shape) == 4 else labels
    dice_pa = dice_score(pred_pa, label_pa)
    return dice_pa


def match_name_keywords(n, name_keywords):
    """Check if name matches any keywords."""
    for b in name_keywords:
        if b in n:
            return True
    return False


def get_logger(filename, verbosity=1, name=None):
    """Create logger for training."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class DataPrefetcher:
    """Prefetch data to GPU for faster training."""
    
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            # 2D data: [B, 3, H, W]
            self.next_batch['image'] = torch.from_numpy(
                self.next_batch['image']
            ).cuda(non_blocking=True).float()
            self.next_batch['label'] = torch.from_numpy(
                self.next_batch['label']
            ).cuda(non_blocking=True).float()
            self.next_batch['weight'] = torch.from_numpy(
                self.next_batch['weight']
            ).cuda(non_blocking=True).float()
            self.next_batch['task_id'] = torch.from_numpy(
                self.next_batch['task_id']
            ).cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch['image'].record_stream(torch.cuda.current_stream())
            batch['label'].record_stream(torch.cuda.current_stream())
            batch['weight'].record_stream(torch.cuda.current_stream())
            batch['task_id'].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch


def main():
    """Main training function."""
    parser = get_arguments()
    print(parser)
    os.environ["OMP_NUM_THREADS"] = "1"

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        
        # Create snapshot directory
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
        
        # Setup logger
        logger = get_logger(os.path.join(args.snapshot_dir, 'log'))
        logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        
        # GPU setup
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        if args.gpu != 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # Parse input size - 2D
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        # Dynamic head configuration
        dep, wid = map(int, args.dyn_head_dep_wid.split(','))
        dyn_head_dep_wid = (dep, wid)

        # Set random seeds
        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True

        # Create model
        model = MOTS_model(
            args, 
            norm_cfg='IN', 
            activation_cfg='relu', 
            num_classes=args.num_classes,
            weight_std=False, 
            deep_supervision=False, 
            res_depth=args.res_depth, 
            dyn_head_dep_wid=dyn_head_dep_wid
        )
        print(model)
        print_model_parm_nums(model)
        logger.info(print_model_parm_nums(model))

        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        # Setup optimizer with different learning rates
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters()
                          if not match_name_keywords(n, ['transformer']) and p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if match_name_keywords(n, ['transformer']) and p.requires_grad],
                "lr": args.learning_rate * args.lr_tr_mult,
            }
        ]
        logger.info([f"Param_dicts info. {len(param_dicts[i]['params'])}_(lr:{param_dicts[i]['lr']})" 
                    for i in range(len(param_dicts))])
        
        if args.optimizer == 'adamw':
            logger.info("Using AdamW optimizer!")
            optimizer = torch.optim.AdamW(param_dicts, args.learning_rate, 
                                         weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            logger.info("Using SGD optimizer!")
            optimizer = torch.optim.SGD(param_dicts, args.learning_rate, 
                                       weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            logger.info("No optimizer defined!")

        # Mixed precision training
        if args.FP16 and HAS_APEX:
            logger.info("Using FP16 mixed precision training")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        elif args.FP16 and not HAS_APEX:
            logger.warning("FP16 requested but APEX not available, using FP32")
            args.FP16 = False

        # Distributed training
        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # Resume from checkpoint
        to_restore = {"epoch": 0}
        utils.restart_from_checkpoint(
            os.path.join(args.snapshot_dir, "checkpoint.pth"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
        )

        # Loss functions
        loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
        loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, 
                                       ignore_index=args.ignore_label).to(device)

        # Create training dataset and loader
        trainloader, train_sampler = engine.get_train_loader(
            MOTSDataSet(
                args.data_dir, 
                args.train_list, 
                max_iters=args.itrs_each_epoch * args.batch_size,
                crop_size=input_size, 
                scale=args.random_scale, 
                mirror=args.random_mirror,
                edge_weight=args.edge_weight
            ), 
            collate_fn=my_collate,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Training loop
        all_tr_loss = []
        start_epoch = to_restore["epoch"]
        
        for epoch in range(start_epoch, args.num_epochs):
            start_time = time.time()
            
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args)

            # Use data prefetcher
            prefetcher = DataPrefetcher(trainloader)
            batch = prefetcher.next()
            
            batch_idx = 0
            while batch is not None:
                images = batch['image']
                labels = batch['label']
                weights = batch['weight']
                task_ids = batch['task_id']

                optimizer.zero_grad()
                
                # Forward pass
                preds = model(images, task_ids)
                del images
                
                # Select prediction for each sample's task
                if args.using_transformer:
                    N_b, N_q, N_c, N_h, N_w = preds.shape
                    preds_convert = torch.zeros(size=(N_b, N_c, N_h, N_w)).cuda()

                    for i_b in range(N_b):
                        preds_convert[i_b] = preds[i_b, int(task_ids[i_b])]
                else:
                    preds_convert = preds

                # Compute losses
                term_seg_Dice = loss_seg_DICE.forward(preds_convert, labels)
                term_seg_BCE = loss_seg_CE.forward(preds_convert, labels, weights)
                term_all = term_seg_Dice + term_seg_BCE

                # Reduce across GPUs
                reduce_all = engine.all_reduce_tensor(term_all)

                # Backward pass
                if args.FP16 and HAS_APEX:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                epoch_loss.append(float(reduce_all))
                del labels, weights

                batch = prefetcher.next()
                batch_idx += 1

            epoch_loss = np.mean(epoch_loss)
            all_tr_loss.append(epoch_loss)
            end_time = time.time()

            if args.local_rank == 0:
                logger.info('Epoch {}: lr1={:.4e}, lr2={:.4e}, loss={:.4f}, time={:.0f}s'.format(
                    epoch, 
                    optimizer.param_groups[0]['lr'], 
                    optimizer.param_groups[1]['lr'], 
                    epoch_loss, 
                    end_time - start_time
                ))

            # Save checkpoint
            if args.local_rank == 0 and epoch % 10 == 0:
                save_dict = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1
                }
                if args.FP16 and HAS_APEX:
                    save_dict['amp'] = amp.state_dict()
                torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))
                logger.info(f"Checkpoint saved at epoch {epoch}")

        end = timeit.default_timer()
        logger.info(f"Training completed in {end - start:.0f} seconds")


if __name__ == '__main__':
    main()
