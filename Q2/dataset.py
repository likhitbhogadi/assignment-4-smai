import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

class CIFAR10ColorizationDataset(Dataset):
    """
    Custom Dataset for CIFAR-10 colorization.
    Input: Grayscale image
    Target: 24-class map of color centroids (PRE-CALCULATED)
    
    The heavy L2 distance calculation is done ONCE during initialization.
    """
    def __init__(self, data, centroids_path="color_centroids.npy"):
        self.data = data
        try:
            # Load centroids
            self.centroids = np.load(centroids_path).astype(np.float32)
        except FileNotFoundError:
            print("Error: 'color_centroids.npy' not found.")
            print("Please ensure the centroid file is in the correct directory.")
            raise
            
        # 1. Transform for input grayscale image
        self.gray_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)) # Normalize grayscale
        ])
        
        # 2. PRE-CALCULATE ALL TARGET CLASS MAPS (The slow step, done once)
        print(f"Pre-calculating color class maps for {len(self.data)} images...")
        self.target_maps = []
        
        # Use tqdm to show the progress of this one-time heavy calculation
        for img_pil, _ in tqdm(self.data, desc="Preprocessing Targets"):
            # Convert PIL image to numpy array (H, W, C)
            rgb_np = np.array(img_pil).astype(np.float32)
            
            # Reshape for distance calculation: (H*W, C)
            h, w, c = rgb_np.shape
            rgb_flat = rgb_np.reshape(-1, c) # Shape: (1024, 3)
            
            # Calculate L2 distance from each pixel to each centroid
            # Formula: dist_sq = ||pixel - centroid||^2
            dists_sq = np.sum((rgb_flat[:, None, :] - self.centroids[None, :, :])**2, axis=2)
            
            # Find the index of the nearest centroid for each pixel
            class_map_flat = np.argmin(dists_sq, axis=1)
            target_map = class_map_flat.reshape(h, w)
            
            # Store as LongTensor (required for CrossEntropyLoss)
            self.target_maps.append(torch.from_numpy(target_map).long())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve raw image (for grayscale input) and pre-calculated target (fast)
        img_pil = self.data[idx][0]
        
        # 1. Grayscale Input Tensor 
        gray_tensor = self.gray_transform(img_pil)
        
        # 2. Retrieve Pre-calculated Target Class Map 
        target_map = self.target_maps[idx]
        
        return gray_tensor, target_map

def get_dataloaders(batch_size=64, seed=42, num_workers=1):
    """
    Loads CIFAR-10, creates Datasets, and returns DataLoaders.
    """
    # Load raw CIFAR-10 data (PIL images)
    train_data = CIFAR10(root="./data", train=True, download=True)
    test_data = CIFAR10(root="./data", train=False, download=True)
    
    # Create custom datasets (The long pre-calculation happens here)
    train_dataset = CIFAR10ColorizationDataset(train_data)
    val_dataset = CIFAR10ColorizationDataset(test_data)
    
    # Create DataLoaders
    # For reproducibility we create a Generator and seed worker RNGs.
    def seed_worker(worker_id):
        # Seed each worker with a combination of the base seed and the worker_id
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        import random as _random
        _random.seed(worker_seed)

    g = torch.Generator()
    try:
        g.manual_seed(int(seed))
    except Exception:
        # If generator doesn't accept seed type, ignore
        pass

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    print(f"DataLoader Ready. Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}.")
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataloader
    print("Testing DataLoader...")
    
    # This call will initiate the slow preprocessing loop first
    train_loader, val_loader = get_dataloaders(batch_size=4) 
    
    # This next line will run quickly once preprocessing is complete
    gray_batch, target_batch = next(iter(train_loader)) 
    
    print(f"\nGrayscale input batch shape: {gray_batch.shape}")
    print(f"Target class map batch shape: {target_batch.shape}")
    print(f"Target min/max: {target_batch.min()} / {target_batch.max()}")