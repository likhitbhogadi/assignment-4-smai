import torch
import wandb
import numpy as np

def class_to_rgb(class_map_tensor, centroids_tensor):
    """
    Converts a batch of class maps to RGB images.
    
    Args:
        class_map_tensor (torch.Tensor): Batch of class maps, shape [B, H, W]
        centroids_tensor (torch.Tensor): Color centroids, shape [K, 3]
        
    Returns:
        torch.Tensor: Batch of RGB images, shape [B, 3, H, W], dtype uint8
    """
    # Ensure tensors are on the same device (or CPU)
    class_map_cpu = class_map_tensor.cpu()
    centroids_cpu = centroids_tensor.cpu()
    
    # Use advanced indexing to map class indices to RGB values
    # centroids_cpu[class_map_cpu] -> [B, H, W, 3]
    rgb_images = centroids_cpu[class_map_cpu]
    
    # Permute to [B, C, H, W] for logging
    rgb_images = rgb_images.permute(0, 3, 1, 2).to(torch.uint8)
    
    return rgb_images

def log_colorization_examples(gray_batch, target_map_batch, pred_map_batch, centroids_tensor, num_examples=10):
    """
    Logs 10 example colorizations to wandb[cite: 224].
    
    Args:
        gray_batch (torch.Tensor): [B, 1, H, W], normalized
        target_map_batch (torch.Tensor): [B, H, W], class indices
        pred_map_batch (torch.Tensor): [B, H, W], class indices
        centroids_tensor (torch.Tensor): [K, 3]
        num_examples (int): Number of images to log
    """
    
    # Take the first N examples
    gray_in = gray_batch[:num_examples]
    target_map = target_map_batch[:num_examples]
    pred_map = pred_map_batch[:num_examples]
    
    # --- 1. Convert Grayscale Input ---
    # Un-normalize from [-1, 1] to [0, 255]
    gray_unnorm = ((gray_in * 0.5) + 0.5) * 255.0
    gray_unnorm = gray_unnorm.clamp(0, 255).to(torch.uint8)
    # Repeat channels to make [B, 3, H, W]
    gray_rgb = gray_unnorm.cpu().repeat(1, 3, 1, 1)
    
    # --- 2. Convert Predicted Map to RGB ---
    pred_rgb = class_to_rgb(pred_map, centroids_tensor)
    
    # --- 3. Convert Ground-Truth Map to RGB ---
    target_rgb = class_to_rgb(target_map, centroids_tensor)
    
    # --- 4. Create wandb.Image objects ---
    example_images = []
    for i in range(num_examples):
        # caption: "Input | Prediction | Ground Truth"
        example_images.append(wandb.Image(
            torch.cat([gray_rgb[i], pred_rgb[i], target_rgb[i]], dim=2),
            caption=f"Example {i+1}: Input | Predicted | Ground Truth"
        ))
        
    return example_images