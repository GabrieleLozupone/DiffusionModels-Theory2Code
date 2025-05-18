import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import diffusers # For UNet2DModel
from dataclasses import dataclass
from torch.amp import GradScaler # Added for mixed precision
from tqdm import tqdm # Added for progress bar
from src.models.unet import UNet

# Assuming src.diffusion.DiffusionProcess is available as in the notebook
# If not, its definition would be needed or it should be in PYTHONPATH
try:
    from src.diffusion import DiffusionProcess
    from src.training import train_step, validation_step # Added import
except ImportError:
    print("Error: src.diffusion.DiffusionProcess or src.training not found. Make sure it\'s in your PYTHONPATH or src directory.")
    exit(1)

# --- Model Definition for Diffusers class ---
@dataclass
class ModelConfig:
    image_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    # UNet specific parameters can be added here if they need to be configurable
    layers_per_block: int = 2
    block_out_channels: tuple = (64, 128, 256, 512)
    down_block_types: tuple = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    )
    up_block_types: tuple = (
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    
# --- Model params for Our Unet ---
unet_args = {
    'input_channel': 3,
    'base_channel': 128,
    'channel_multiplier': [1, 2, 3, 4],
    'num_residual_blocks_of_a_block': 2,
    'attention_resolutions': [3, 4],
    'num_heads': 1,
    'head_channel': -1,
    'use_new_attention_order': False,
    'dropout': 0.1,
    'dims': 2
}


class CustomUNetModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = diffusers.UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types
        )

    def forward(self, x, t):
        return self.model(x, t, return_dict=False)[0]

# --- Helper Functions ---
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def save_generated_images(samples, output_dir: Path, epoch: int, n_to_save=16, grid_size=(4,4)):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*2.5, grid_size[0]*2.5))
    axes = axes.flatten()
    for i in range(min(n_to_save, samples.shape[0], len(axes))):
        ax = axes[i]
        img = (samples[i].cpu().permute(1, 2, 0) + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        ax.imshow(img.clamp(0, 1))
        ax.axis('off')
    for j in range(i + 1, len(axes)): # Turn off unused subplots
        axes[j].axis('off')

    plt.suptitle(f"Generated Samples at Epoch {epoch + 1}")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    save_path = output_dir / f"epoch_{epoch+1:04d}_samples.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {min(n_to_save, samples.shape[0])} generated samples to {save_path}")

def main(args):
    # --- Setup ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id is not None else ""
    print(f"Using GPU ID: {args.gpu_id}")
    device = torch.device(f"cuda" if torch.cuda.is_available() and args.gpu_id is not None else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    experiment_base_dir = Path(args.output_dir) / args.experiment_name
    image_save_dir = experiment_base_dir / "generated_images"
    model_save_dir = Path(args.model_dir) / args.experiment_name
    
    experiment_base_dir.mkdir(parents=True, exist_ok=True)
    image_save_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        full_train_dataset = datasets.CelebA(root=args.data_root, split="train", download=args.download_data, transform=transform)
        full_test_dataset = datasets.CelebA(root=args.data_root, split="test", download=args.download_data, transform=transform)
    except Exception as e:
        print(f"Error loading CelebA dataset: {e}")
        print(f"Please ensure the dataset is available at {args.data_root} or set --download_data.")
        return

    # Create train subset only if train_subset_size is specified
    if args.train_subset_size > 0:
        train_subset_indices = np.random.choice(len(full_train_dataset), min(args.train_subset_size, len(full_train_dataset)), replace=False)
        train_dataset = Subset(full_train_dataset, train_subset_indices)
    else:
        train_dataset = full_train_dataset
    
    # Create validation subset only if val_subset_size is specified
    if args.val_subset_size > 0:
        val_subset_indices = np.random.choice(len(full_test_dataset), min(args.val_subset_size, len(full_test_dataset)), replace=False)
        val_dataset = Subset(full_test_dataset, val_subset_indices)
    else:
        val_dataset = full_test_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Training subset size: {len(train_dataset)}, Validation subset size: {len(val_dataset)}")

    # --- Diffusion Process ---
    beta_schedule = linear_beta_schedule(args.timesteps, args.beta_start, args.beta_end).to(device)
    diffusion = DiffusionProcess(beta_schedule)

    # --- Model ---
    if args.use_diffusers_model:
        print("Using diffusers UNet2DModel.")
        model_config = ModelConfig(image_size=args.image_size)
        model = CustomUNetModel(model_config).to(device)
    else:
        print("Using our Unet model.")
        model = UNet(**unet_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize GradScaler for mixed precision training
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        scaler = GradScaler()
        print("Using mixed precision training.")
    elif args.mixed_precision and device.type != 'cuda':
        print("Warning: Mixed precision requested but CUDA is not available. Using full precision.")
    else:
        print("Using full precision training.")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Checkpointing & Resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    # Save in the experiment directory
    checkpoint_dir = experiment_base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_last.pth"

    if args.resume:
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Handle older checkpoints
                # Restore args if needed, or check for consistency
                # loaded_args = checkpoint.get('args')
                # if loaded_args:
                #     print("Loaded args from checkpoint:", loaded_args)
                print(f"Resumed training from epoch {start_epoch}. Best val loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Could not load checkpoint: {e}. Starting from scratch.")
                args.resume = False # Force start from scratch if checkpoint is faulty
        else:
            print("No checkpoint found. Starting training from scratch.")
            args.resume = False

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_epoch = 0
        # Wrap train_loader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_idx, batch_data in enumerate(progress_bar): # Changed variable name
            # images = images.to(device) # Moved to train_step
            # optimizer.zero_grad() # Moved to train_step
            
            # t = torch.randint(0, args.timesteps, (images.shape[0],), device=device).long() # Moved to train_step
            
            loss = train_step(model, optimizer, batch_data, diffusion, device, scaler) # Use train_step
            
            train_loss_epoch += loss # train_step returns loss.item()
            
            # Update tqdm progress bar
            progress_bar.set_postfix(loss=f"{loss:.4f}")
            

        avg_train_loss = train_loss_epoch / len(train_loader)
        # Clear the tqdm progress bar for the epoch
        progress_bar.close()
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation & Image Saving ---
        if (epoch + 1) % args.eval_every_n_epochs == 0 or epoch == 0:
            avg_val_loss = validation_step(model, val_loader, diffusion, device) # Use validation_step
            print(f"Epoch [{epoch+1}/{args.epochs}], Average Validation Loss: {avg_val_loss:.4f}")

            # Save Generated Images
            if args.num_val_samples_to_generate > 0:
                generated_samples = diffusion.sample(model, 
                                                     n_samples=args.num_val_samples_to_generate, 
                                                     device=device, 
                                                     size=(3, args.image_size, args.image_size))
                save_generated_images(generated_samples, image_save_dir, epoch, 
                                      n_to_save=args.num_val_samples_to_generate, 
                                      grid_size=args.val_sample_grid_size)

            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = checkpoint_dir / "model_best.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        
        # --- Save Checkpoint (Last Model for resuming) ---
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'args': vars(args) # Save args for reference
        }
        torch.save(checkpoint, checkpoint_path)
        # print(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")


    print("Training complete.")
    final_model_path = model_save_dir / "model_final_epoch.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model state at epoch {args.epochs} to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Model on CelebA")
    
    # Paths and Naming
    parser.add_argument("--experiment_name", type=str, default="celeba_64x64", help="Name for the experiment run")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of CelebA dataset")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Base directory for saving outputs (images, logs)")
    parser.add_argument("--model_dir", type=str, default="celeba_models", help="Base directory for saving model checkpoints and final models")
    parser.add_argument("--download_data", action="store_true", help="Download CelebA dataset if not found")

    # Training Control
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=None, help="Specific GPU ID to use (e.g., 0, 1). Uses any available if not set.")

    # Model & Diffusion Parameters
    parser.add_argument("--image_size", type=int, default=64, help="Size to resize images to (image_size x image_size)")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta start for linear schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end for linear schedule")
    parser.add_argument("--use_diffusers_model", action="store_true", help="Use diffusers UNet2DModel instead of custom UNet")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training (requires CUDA)")
    
    # Dataset Size
    parser.add_argument("--train_subset_size", type=int, default=0, help="Number of samples for training subset")
    parser.add_argument("--val_subset_size", type=int, default=0, help="Number of samples for validation subset")

    # Evaluation and Logging
    parser.add_argument("--eval_every_n_epochs", type=int, default=5, help="Frequency of evaluation and image saving (in epochs)")
    parser.add_argument("--log_interval", type=int, default=50, help="Frequency of logging training loss (in steps/batches)")
    parser.add_argument("--num_val_samples_to_generate", type=int, default=16, help="Number of samples to generate during validation")
    parser.add_argument("--val_sample_grid_size", type=int, nargs=2, default=[4,4], metavar=('ROWS', 'COLS'), help="Grid size (rows cols) for saving validation samples")


    args = parser.parse_args()
    
    # Ensure grid size matches num_val_samples_to_generate if possible, or adjust num_val_samples_to_generate
    if args.num_val_samples_to_generate > args.val_sample_grid_size[0] * args.val_sample_grid_size[1]:
        print(f"Warning: num_val_samples_to_generate ({args.num_val_samples_to_generate}) is greater than grid capacity ({args.val_sample_grid_size[0] * args.val_sample_grid_size[1]}). Clamping to grid capacity.")
        args.num_val_samples_to_generate = args.val_sample_grid_size[0] * args.val_sample_grid_size[1]
        
    main(args)
