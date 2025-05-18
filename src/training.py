"""
Training module for diffusion models.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler  # Updated import path for mixed precision components

def train_step(model, optimizer, batch, diffusion, device, scaler=None):
    """
    Execute a single training step.
    
    Args:
        model: The noise prediction model
        optimizer: The optimizer
        batch: Batch of data
        diffusion: The diffusion process
        device: Device to run training on
        scaler: GradScaler for mixed precision training
    
    Returns:
        loss: The training loss for this step
    """
    optimizer.zero_grad()
    
    # CelebA returns (image, target) where target is a multi-label tensor
    # MNIST returns (image, label) where label is a class index
    if isinstance(batch, list) or isinstance(batch, tuple):
        x0 = batch[0].to(device)  # Get images, ignore labels
    else:
        x0 = batch.to(device)
    
    batch_size = x0.shape[0]

    # Sample timesteps uniformly
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

    # Get noisy images and target noise
    noisy_images, noise = diffusion.corrupt_image(x0, t)

    # Use mixed precision if scaler is provided
    if scaler is not None:
        with autocast(device_type=device.type):
            # Predict noise
            predicted_noise = model(noisy_images, t)
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # Use scaler for backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard full precision training
        predicted_noise = model(noisy_images, t)
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()

    return loss.item()

def validation_step(model, val_loader, diffusion, device):
    """
    Execute validation on the validation set.
    
    Args:
        model: The noise prediction model
        val_loader: DataLoader for validation data
        diffusion: The diffusion process
        device: Device to run validation on
    
    Returns:
        val_loss: The average validation loss
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # CelebA returns (image, target) where target is a multi-label tensor
            # MNIST returns (image, label) where label is a class index
            if isinstance(batch, list) or isinstance(batch, tuple):
                x0 = batch[0].to(device)  # Get images, ignore labels
            else:
                x0 = batch.to(device)
                
            batch_size = x0.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
            noisy_images, noise = diffusion.corrupt_image(x0, t)
            predicted_noise = model(noisy_images, t)
            val_loss += torch.nn.functional.mse_loss(predicted_noise, noise).item()
    model.train()
    return val_loss / len(val_loader)

def train_diffusion_model(model, train_dataset, val_dataset, diffusion, device,
                         batch_size=64, epochs=100, eval_every_n_epochs=5, lr=1e-4,
                         model_dir='models', model_name='diffusion_model', num_workers=10,
                         mixed_precision=True):
    """
    Train a diffusion model.
    
    Args:
        model: The noise prediction model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        diffusion: The diffusion process
        device: Device to run training on
        batch_size: Batch size for training
        epochs: Number of training epochs
        eval_every_n_epochs: Frequency of evaluation and sample generation
        lr: Learning rate
        model_dir: Directory to save models
        model_name: Name of the model file (without extension)
        num_workers: Number of workers for data loading
        mixed_precision: Whether to use mixed precision training
    
    Returns:
        model: The trained model
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize GradScaler for mixed precision training
    if mixed_precision and device.type == 'cuda':
        scaler = GradScaler()
    else:
        scaler = None
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            loss = train_step(model, optimizer, batch, diffusion, device, scaler)
            train_loss += loss

            progress_bar.set_postfix({'loss': loss})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = validation_step(model, val_loader, diffusion, device)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create the directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            # Save the model state
            model_path = os.path.join(model_dir, f'{model_name}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}')
            
        # Save the last model state
        last_model_path = os.path.join(model_dir, f'{model_name}_last.pth')
        torch.save(model.state_dict(), last_model_path)
        print(f'Saved last model at epoch {epoch+1}')

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Generate samples every n epochs
        if (epoch + 1) % eval_every_n_epochs == 0 or epoch == 0:
            samples = diffusion.sample(model, n_samples=4, device=device, size=train_dataset[0][0].shape)
            plt.figure(figsize=(8, 2))
            for i in range(4):
                plt.subplot(1, 4, i+1)
                # Convert from [-1, 1] to [0, 1] range
                img = (samples[i].cpu().permute(1, 2, 0) + 1) / 2
                plt.imshow(img.clamp(0, 1))
                plt.axis('off')
            plt.suptitle(f'Generated Samples - Epoch {epoch+1}')
            plt.show()
            
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    return model

def visualize_forward_diffusion(diffusion, images, device, num_timesteps=5, steps=None):
    """
    Visualize the forward diffusion process on a batch of images.
    
    Args:
        diffusion: DiffusionProcess instance
        images: Tensor of images to visualize
        device: Device to run the diffusion on
        num_timesteps: Number of timesteps to show
        steps: Specific timesteps to visualize
    """
    if steps is None:
        timesteps = torch.linspace(0, diffusion.timesteps-1, num_timesteps).long()
    else:
        timesteps = torch.tensor(steps).long()
        num_timesteps = len(steps)

    fig, axes = plt.subplots(len(images), num_timesteps + 1, figsize=(2*(num_timesteps + 1), 2*len(images)))    
    
    for i, image in enumerate(images):
        # Show original image
        if len(image.shape) == 3 and image.shape[0] == 3:  # RGB image
            # Convert from [-1, 1] to [0, 1] range for visualization
            img_viz = (image.cpu().permute(1, 2, 0) + 1) / 2
            axes[i, 0].imshow(img_viz.clamp(0, 1))
        else:  # Grayscale image
            axes[i, 0].imshow(image.squeeze().cpu(), cmap='gray')
            
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Show noisy versions
        for j, t in enumerate(timesteps, 1):
            # Move everything to the same device
            img = image.unsqueeze(0).to(device)
            t_tensor = torch.tensor([t], device=device)
            noisy_image, _ = diffusion.corrupt_image(img, t_tensor)
            
            if len(image.shape) == 3 and image.shape[0] == 3:  # RGB image
                # Convert from [-1, 1] to [0, 1] range for visualization
                noisy_viz = (noisy_image[0].cpu().permute(1, 2, 0) + 1) / 2
                axes[i, j].imshow(noisy_viz.clamp(0, 1))
            else:  # Grayscale image
                axes[i, j].imshow(noisy_image.squeeze().cpu(), cmap='gray')
                
            axes[i, j].set_title(f't={t.item()}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
