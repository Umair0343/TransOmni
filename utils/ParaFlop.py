"""
TransOmni - Model Parameter Counter
"""


def print_model_parm_nums(model):
    """Print and return number of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        String with parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    msg = f"Total params: {total:,} | Trainable: {trainable:,}"
    print(msg)
    return msg
