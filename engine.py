"""
TransOmni - Training Engine
Provides distributed training support and data loading utilities

Adapted from TransDoDNet's engine for 2D training
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Engine:
    """Training engine for distributed and single-GPU training."""
    
    def __init__(self, custom_parser=None):
        self.parser = custom_parser
        self.distributed = False
        self.local_rank = 0
        self.world_size = 1
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def data_parallel(self, model):
        """Wrap model in DistributedDataParallel if needed.
        
        Args:
            model: PyTorch model
            
        Returns:
            Parallelized model
        """
        if self.distributed:
            model = DistributedDataParallel(
                model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        return model
    
    def get_train_loader(self, dataset, collate_fn=None, batch_size=None, 
                         num_workers=4, shuffle=True, drop_last=True):
        """Create training data loader.
        
        Args:
            dataset: Training dataset
            collate_fn: Custom collate function
            batch_size: Batch size (uses args.batch_size if None)
            num_workers: Number of data loading workers
            shuffle: Shuffle data
            drop_last: Drop last incomplete batch
            
        Returns:
            Tuple of (dataloader, sampler)
        """
        args = self.parser.parse_args() if self.parser else argparse.Namespace()
        
        if batch_size is None:
            batch_size = getattr(args, 'batch_size', 2)
        
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=True
        )
        
        return loader, sampler
    
    def get_val_loader(self, dataset, collate_fn=None, batch_size=1, num_workers=4):
        """Create validation data loader.
        
        Args:
            dataset: Validation dataset
            collate_fn: Custom collate function
            batch_size: Batch size
            num_workers: Number of data loading workers
            
        Returns:
            Tuple of (dataloader, sampler)
        """
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=sampler,
            pin_memory=True
        )
        
        return loader, sampler
    
    def all_reduce_tensor(self, tensor, norm=True):
        """All-reduce tensor across distributed processes.
        
        Args:
            tensor: Input tensor
            norm: Normalize by world size
            
        Returns:
            Reduced tensor
        """
        if self.distributed:
            dist.all_reduce(tensor)
            if norm:
                tensor /= self.world_size
        return tensor
