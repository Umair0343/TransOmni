"""
TransOmni - 2D DeepNorm Coefficients
Adapted from TransDoDNet's deepnorm.py

This file provides coefficients for DeepNorm initialization,
which helps stabilize training of deep transformers.
"""


def get_deepnorm_coefficients(encoder_layers, decoder_layers):
    """Get alpha and beta coefficients for DeepNorm.
    
    DeepNorm (https://arxiv.org/abs/2203.00555) provides a way to
    stabilize training of very deep transformers.
    
    Args:
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        
    Returns:
        Tuple of (encoder_alpha, decoder_alpha, encoder_beta, decoder_beta)
    """
    # Alpha values for residual scaling
    encoder_alpha = (2 * encoder_layers) ** 0.25
    decoder_alpha = (2 * decoder_layers) ** 0.25
    
    # Beta values for weight initialization scaling
    encoder_beta = (8 * encoder_layers) ** -0.25
    decoder_beta = (8 * decoder_layers) ** -0.25
    
    return encoder_alpha, decoder_alpha, encoder_beta, decoder_beta
