import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
import yaml
import random
import os

from model import ColorizationNet
from dataset import get_dataloaders
from utils import log_colorization_examples

def load_config(config_path="config.yaml"):
    """Loads the YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for gray_img, target_map in pbar:
        gray_img, target_map = gray_img.to(device), target_map.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(gray_img) # [B, 24, 32, 32]

        # Loss calculation [cite: 220]
        loss = criterion(outputs, target_map)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(train_loss=loss.item())

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, centroids_tensor, epoch_num):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    
    # Store one batch for visualization
    vis_gray, vis_target = None, None
    
    with torch.no_grad():
        for gray_img, target_map in tqdm(loader, desc="Validation"):
            gray_img, target_map = gray_img.to(device), target_map.to(device)
            
            if vis_gray is None:
                vis_gray, vis_target = gray_img, target_map
            
            # Forward pass
            outputs = model(gray_img)
            
            # Loss
            loss = criterion(outputs, target_map)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader)
    
    # --- Log example images ---
    # Get model predictions for the visualization batch
    vis_outputs = model(vis_gray)
    vis_pred_map = torch.argmax(vis_outputs, dim=1) # [B, H, W]
    
    example_images = log_colorization_examples(
        vis_gray, vis_target, vis_pred_map, centroids_tensor, num_examples=10
    )
    
    wandb.log({
        "epoch": epoch_num,
        "val_loss": avg_loss,
        "colorization_examples": example_images
    })
    
    return avg_loss

def run_training():
    """Main training and evaluation loop, configured by wandb."""
    
    # Load base config from YAML
    base_config = load_config("config.yaml")

    # Initialize W&B
    # W&B will use 'base_config' as defaults.
    # If run by a sweep, it will override these defaults.
    # Optionally allow the YAML config to specify a run_name. If provided,
    # use it as the W&B run name. Otherwise generate a timestamped default.
    run_name = base_config.pop('run_name', None)

    if run_name is None:
        from datetime import datetime
        safe_proj = base_config.get('project_name', 'wandb-project')
        run_name = f"{safe_proj}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    run = wandb.init(
        project=base_config.pop('project_name'), # Use and remove from config
        entity=base_config.pop('entity'),       # Use and remove from config
        name=run_name,                          # Human-friendly run name
        config=base_config                      # Pass the rest as config defaults
    )
    
    # Get the final config from wandb (defaults + sweep overrides)
    config = wandb.config
    
    print("Starting run with config:", config)
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    # -------------------------
    # Deterministic / reproducible setup
    # -------------------------
    # Read seed from config (if provided) or default to 42
    seed = int(config.get('seed', 42))

    # Set PYTHONHASHSEED for deterministic hashing
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms where possible (newer PyTorch)
    # Use warn_only=True because some ops (like CrossEntropyLoss on CUDA) don't have deterministic implementations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only parameter
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Even older versions don't have this API at all
            pass
    
    # Load data (pass seed and optionally num_workers for reproducibility)
    num_workers = int(config.get('num_workers', 1))
    train_loader, val_loader = get_dataloaders(batch_size=config.batch_size, seed=seed, num_workers=num_workers)
    
    # Load centroids
    centroids = np.load("color_centroids.npy")
    centroids_tensor = torch.from_numpy(centroids).to(device)
    
    # Initialize Model
    model = ColorizationNet(
        nic=1,
        nf=config.num_filters_nf,
        nc=24,
        kernel_size=config.kernel_size
    ).to(device)
    
    # Initialize Optimizer
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
        
    # Initialize Loss
    criterion = nn.CrossEntropyLoss()
    
    # Log model graph and gradients. Set log_freq to the length of the training
    # loader so we record gradients/parameters once per epoch instead of every N steps.
    try:
        log_freq = max(1, len(train_loader))
    except Exception:
        log_freq = 100
    wandb.watch(model, criterion, log="all", log_freq=log_freq)
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1} / {config.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device, centroids_tensor, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": train_loss,
        })
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            checkpoint_path = f"best_model_{run.id}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Log to wandb
            wandb.save(checkpoint_path, base_path=".", policy="live")
            print(f"New best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
            
    print("\n--- Training Complete ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Log final best loss for the sweep
    wandb.summary["best_val_loss"] = best_val_loss
    
    run.finish()

if __name__ == "__main__":
    run_training()