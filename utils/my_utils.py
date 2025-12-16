"""
TransOmni - Utility Functions
Adapted from TransDoDNet's utils for training and model management
"""

import os
import torch


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """Restart from checkpoint if it exists.
    
    Args:
        ckp_path: Path to checkpoint file
        run_variables: Dict of variables to restore (e.g., epoch)
        **kwargs: Named parameters to restore (e.g., state_dict=model)
    """
    if not os.path.isfile(ckp_path):
        print(f"No checkpoint found at {ckp_path}")
        return
    
    print(f"Loading checkpoint from {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location='cpu')
    
    for key, value in kwargs.items():
        if key in checkpoint:
            if hasattr(value, 'load_state_dict'):
                try:
                    value.load_state_dict(checkpoint[key])
                except Exception as e:
                    print(f"Warning: Could not load {key}: {e}")
            else:
                kwargs[key] = checkpoint[key]
        else:
            print(f"Key {key} not found in checkpoint")
    
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
            else:
                print(f"Variable {var_name} not found in checkpoint")
    
    print(f"Successfully loaded checkpoint from {ckp_path}")


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save checkpoint to file.
    
    Args:
        state: Dict containing state to save
        filename: Output filename
    """
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")
