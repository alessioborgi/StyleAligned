"""
adain.py

This file contains the implementation of the Adaptive Instance Normalization (AdaIN) function.

Authors:
- Alessio Borgi (alessioborgi3@gmail.com)

Created on: July 6, 2024
"""

import torch

def AdaIN(feature_tensor: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Instance Normalization (AdaIN) function.
    
    Args:
        feature_tensor (torch.Tensor): The input feature tensor to be normalized and styled.
        
    Returns:
        torch.Tensor: The tensor after applying AdaIN.
    """
    # Calculate the mean and standard deviation of the input feature tensor along the -2 dimension.
    mean_feature = feature_tensor.mean(dim=-2, keepdims=True)
    stddev_feature = (feature_tensor.var(dim=-2, keepdims=True) + 1e-5).sqrt()

    # Get batch size.
    batch_size = feature_tensor.shape[0]

    # Step 1: Expanding the Mean and Standard Deviation for Style.
    mean_feature_first = mean_feature[0]  # Mean of the first element in the batch.
    mean_feature_middle = mean_feature[batch_size // 2]  # Mean of the middle element in the batch.

    stacked_mean_style = torch.stack((mean_feature_first, mean_feature_middle))  # Stack the first and middle means.
    expanded_mean_style_unsqueezed = stacked_mean_style.unsqueeze(1)  # Add a new dimension.
    expanded_mean_style = expanded_mean_style_unsqueezed.expand(2, batch_size // 2, *mean_feature.shape[1:])  # Expand to match the batch size.

    stddev_feature_first = stddev_feature[0]  # StdDev of the first element in the batch.
    stddev_feature_middle = stddev_feature[batch_size // 2]  # StdDev of the middle element in the batch.

    stacked_stddev_style = torch.stack((stddev_feature_first, stddev_feature_middle))  # Stack the first and middle stddevs.
    expanded_stddev_style_unsqueezed = stacked_stddev_style.unsqueeze(1)  # Add a new dimension.
    expanded_stddev_style = expanded_stddev_style_unsqueezed.expand(2, batch_size // 2, *stddev_feature.shape[1:])  # Expand to match the batch size.

    # Step 2: Applying Style Mean and Standard Deviation.
    expanded_mean_style_reshaped = expanded_mean_style.reshape(*feature_tensor.shape)  # Reshape to match input tensor shape.
    expanded_stddev_style_reshaped = expanded_stddev_style.reshape(*feature_tensor.shape)  # Reshape to match input tensor shape.

    # Normalize the input features using the calculated mean and standard deviation.
    normalized_feature = (feature_tensor - mean_feature) / stddev_feature

    # Apply the style mean and standard deviation to the normalized features.
    styled_feature = normalized_feature * expanded_stddev_style_reshaped + expanded_mean_style_reshaped

    return styled_feature