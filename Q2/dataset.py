import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import numpy as np

class CIFAR10ColorizationDataset(Dataset):
    """
    Custom Dataset for CIFAR-10 colorization.
    Input: Grayscale image
    Target: 24-class map of color centroids
    """
    def __init__(self, data, centroids_path="color_centroids.npy", train=True):
        self.data = data
        try:
            # Load centroids and convert to numpy
            self.centroids = np.load(centroids_path).astype(np.float32)
        except FileNotFoundError:
            print("Error: 'color_centroids.npy' not found.")
            print("Please run 'create_centroids.py' and replace the placeholder with the assignment file.")
            raise
            
        # Transform for input grayscale image
        self.gray_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)) # Normalize grayscale
        ])
        
        # Store original PIL images for target creation
        self.pil_images = [img for img, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the original PIL image (32x32 RGB)
        img_pil = self.pil_images[idx]
        
        # 1. Create Grayscale Input Tensor [B, 1, 32, 32]
        gray_tensor = self.gray_transform(img_pil)
        
        # 2. Create Target Class Map [32, 32]
        # Convert PIL image to numpy array (H, W, C)
        rgb_np = np.array(img_pil).astype(np.float32)
        
        # Reshape for distance calculation: (H*W, C)
        h, w, c = rgb_np.shape
        rgb_flat = rgb_np.reshape(-1, c) # Shape: (1024, 3)
        
        # Calculate L2 distance from each pixel to each centroid
        # (1024, 1, 3) - (1, 24, 3) -> (1024, 24, 3)
        dists_sq = np.sum((rgb_flat[:, None, :] - self.centroids[None, :, :])**2, axis=2)
        
        # Find the index of the nearest centroid for each pixel [cite: 130]
        # Shape: (1024,)
        class_map_flat = np.argmin(dists_sq, axis=1)
        
        # Reshape back to (H, W)
        target_map = class_map_flat.reshape(h, w)
        
        # Return input and target
        # Target must be LongTensor for CrossEntropyLoss
        return gray_tensor, torch.from_numpy(target_map).long()

def get_dataloaders(batch_size=64):
    """
    Loads CIFAR-10, creates Datasets, and returns DataLoaders.
    """
    # Load raw CIFAR-10 data (PIL images)
    train_data = CIFAR10(root="./data", train=True, download=True)
    test_data = CIFAR10(root="./data", train=False, download=True)
    
    # Create custom datasets
    train_dataset = CIFAR10ColorizationDataset(train_data, train=True)
    val_dataset = CIFAR10ColorizationDataset(test_data, train=False)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Loaded {len(train_dataset)} training images.")
    print(f"Loaded {len(val_dataset)} validation images.")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataloader
    print("Testing DataLoader...")
    train_loader, val_loader = get_dataloaders(batch_size=4)
    gray_batch, target_batch = next(iter(train_loader))
    
    print(f"Grayscale input batch shape: {gray_batch.shape}")
    print(f"Target class map batch shape: {target_batch.shape}")
    print(f"Input dtype: {gray_batch.dtype}")
    print(f"Target dtype: {target_batch.dtype}")
    print(f"Target min/max: {target_batch.min()} / {target_batch.max()}")
    
    # Expected output:
    # Grayscale input batch shape: torch.Size([4, 1, 32, 32])
    # Target class map batch shape: torch.Size([4, 32, 32])
    # Input dtype: torch.float32
    # Target dtype: torch.int64
    # Target min/max: 0 / 23 (approx)