import torch
from tqdm import tqdm

class DiffusionProcess:
    """
    Base diffusion process class that implements DDPM (Denoising Diffusion Probabilistic Models).
    """
    def __init__(self, beta_schedule):
        """
        Initialize the DiffusionProcess class.
        Args:
            beta_schedule (torch.Tensor): The variance schedule (beta values) for the diffusion process.
        """
        self.beta_schedule = beta_schedule
        self.alpha = 1 - beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.timesteps = len(beta_schedule)
        self.device = beta_schedule.device

    def corrupt_image(self, batch, t):
        """
        Get the corrupted image at timestep t.
        Args:
            batch (torch.Tensor): A batch of images (e.g., MNIST images).
            t (torch.Tensor): Timesteps for each image in the batch.
        Returns:
            torch.Tensor: The corrupted images at timestep t.
        """
        # Ensure batch and t are on the same device as beta_schedule
        batch = batch.to(self.device)
        t = t.to(self.device)
        
        # Properly reshape alpha_bar for broadcasting
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)

        noise = torch.randn_like(batch)  # Generate Gaussian noise
        corrupted_batch = sqrt_alpha_bar_t * batch + sqrt_one_minus_alpha_bar_t * noise
        return corrupted_batch, noise  # Return both corrupted image and noise for training

    def sample(self, model, n_samples, device, size=(1, 32, 32)):
        """
        Sample new images using the trained model.
        Args:
            model: The trained noise prediction model
            n_samples: Number of images to generate
            device: Device to run the sampling on
            size: Size of each image (channels, height, width)
        Returns:
            Generated images
        """
        model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn((n_samples, *size)).to(device)

            for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling"):
                # Create a batch of timesteps
                timesteps = torch.full((n_samples,), t, device=device, dtype=torch.long)

                # Predict the noise
                predicted_noise = model(x, timesteps)

                # Calculate parameters for reverse process
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.beta_schedule[t]

                # No noise at timestep 0
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                # Compute the mean for reverse process
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise

        model.train()
        return x
    
    def predict_x0_from_xt(self, xt, t, predicted_noise):
        """
        Predict x0 from xt and the predicted noise.
        Args:
            xt (torch.Tensor): The noisy image at timestep t.
            t (torch.Tensor): The timestep.
            predicted_noise (torch.Tensor): The predicted noise.
        Returns:
            torch.Tensor: The predicted original image x0.
        """
        # Reshape alpha_bar for broadcasting
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        
        # Predict x0 using the equation: x0 = (xt - sqrt(1-α̅t)ε)/sqrt(α̅t)
        predicted_x0 = (xt - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        
        return predicted_x0


class DDIM(DiffusionProcess):
    """
    Denoising Diffusion Implicit Models (DDIM) implementation.
    Extends the DiffusionProcess class to provide accelerated sampling and image encoding.
    """
    
    def ddim_step(self, model_output, timestep, next_timestep, sample, eta=0.0):
        """
        Perform a single DDIM sampling step.
        
        Args:
            model_output (torch.Tensor): The predicted noise from the model
            timestep (int): The current timestep
            next_timestep (int): The next timestep to move to (lower than current)
            sample (torch.Tensor): The current sample (noisy image at timestep t)
            eta (float): Controls stochasticity (0.0 = deterministic DDIM, 1.0 = DDPM equivalent)
            
        Returns:
            torch.Tensor: The updated sample at the next timestep (t_prev)
        """
        # Get batch size for proper broadcasting
        batch_size = sample.shape[0]
        
        # Create tensors for broadcasting
        timestep_tensor = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
        
        # Predict x0 from xt and the predicted noise
        predicted_x0 = self.predict_x0_from_xt(sample, timestep_tensor, model_output)
        
        # Get the current alpha values
        alpha_bar_t = self.alpha_bar[timestep]
        alpha_bar_prev = self.alpha_bar[next_timestep] if next_timestep >= 0 else torch.tensor(1.0).to(self.device)
        
        # Reshape for broadcasting
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).expand(batch_size, 1, 1, 1)
        
        # Calculate parameters for the reverse process
        beta_t = self.beta_schedule[timestep]
        beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
        
        # Calculate sigma_t based on eta parameter
        sigma_t = eta * torch.sqrt(beta_tilde)
        
        # Calculate direction coefficient 
        # sqrt(1-α̅t-1 - σ²_t) for the noise component
        direction_coef = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2).expand(batch_size, 1, 1, 1)
        
        # Add random noise if eta > 0
        random_noise = torch.randn_like(sample) if eta > 0 and next_timestep > 0 else torch.zeros_like(sample)
        random_noise = random_noise.to(self.device)
        
        # DDIM update rule
        x_t_prev = sqrt_alpha_bar_prev * predicted_x0 + direction_coef * model_output + sigma_t.expand(batch_size, 1, 1, 1) * random_noise
        
        return x_t_prev
    
    def sample_ddim(self, model, n_samples, device, size=(1, 32, 32), 
                   timestep_subset=None, eta=0.0, x_T=None):
        """
        Sample new images using DDIM with controllable stochasticity.
        Args:
            model: The trained noise prediction model
            n_samples: Number of images to generate
            device: Device to run the sampling on
            size: Size of each image (channels, height, width)
            timestep_subset: Subset of timesteps to use for accelerated sampling
                            If None, use all timesteps
            eta: Controls stochasticity (0.0 = deterministic DDIM, 1.0 = DDPM equivalent)
            x_T: Optional starting point for reverse process. If None, random noise is used.
        Returns:
            Generated images
        """
        model.eval()
        with torch.no_grad():
            # Start from provided x_T or pure noise
            if x_T is not None:
                x_t = x_T.to(device)
            else:
                x_t = torch.randn((n_samples, *size)).to(device)
            
            # Define timesteps to use
            if timestep_subset is None:
                timesteps = list(range(self.timesteps - 1, -1, -1))
            else:
                timesteps = sorted(timestep_subset, reverse=True)
            
            for i in tqdm(range(len(timesteps) - 1), desc="DDIM Sampling"):
                # Current and next timestep in the subsequence
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                
                # Create a batch of timesteps
                timestep_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
                
                # Predict the noise
                predicted_noise = model(x_t, timestep_tensor)
                
                # Predict x0 from xt and the predicted noise
                predicted_x0 = self.predict_x0_from_xt(x_t, timestep_tensor, predicted_noise)
                
                # Get the current alpha values
                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
                
                # Reshape for broadcasting - use expand instead of view for scalar tensors
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).expand(n_samples, 1, 1, 1)
                
                # Calculate parameters for reverse process
                beta_t = self.beta_schedule[t]
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                
                # Calculate sigma_t based on eta
                sigma_t = eta * torch.sqrt(beta_tilde)
                
                # Calculate direction coefficient
                # sqrt(1-α̅t-1 - σ²_t) for the noise component
                direction_coef = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2).expand(n_samples, 1, 1, 1)
                
                # Add random noise if eta > 0
                random_noise = torch.randn_like(x_t) if eta > 0 and t_prev > 0 else torch.zeros_like(x_t)
                
                # DDIM update rule
                x_t_prev = sqrt_alpha_bar_prev * predicted_x0 + direction_coef * predicted_noise + sigma_t.expand(n_samples, 1, 1, 1) * random_noise
                
                # Update x_t for next iteration
                x_t = x_t_prev

        model.train()
        return x_t
    
    def sample_ddim_with_intermediates(self, model, n_samples, device, size=(1, 32, 32),
                                      timestep_subset=None, eta=0.0, x_T=None, save_intermediates_at=None):
        """
        Sample new images using DDIM with controllable stochasticity and save intermediate steps.
        
        Args:
            model: The trained noise prediction model
            n_samples: Number of images to generate
            device: Device to run the sampling on
            size: Size of each image (channels, height, width)
            timestep_subset: Subset of timesteps to use for accelerated sampling
                            If None, use all timesteps
            eta: Controls stochasticity (0.0 = deterministic DDIM, 1.0 = DDPM equivalent)
            x_T: Optional starting point for reverse process. If None, random noise is used.
            save_intermediates_at: Indices at which to save intermediate results (within timestep_subset)
            
        Returns:
            Tuple of (generated images, list of intermediate images)
        """
        model.eval()
        intermediates = []
        
        with torch.no_grad():
            # Start from provided x_T or pure noise
            if x_T is not None:
                x_t = x_T.to(device)
            else:
                x_t = torch.randn((n_samples, *size)).to(device)
            
            # Define timesteps to use
            if timestep_subset is None:
                timesteps = list(range(self.timesteps - 1, -1, -1))
            else:
                timesteps = sorted(timestep_subset, reverse=True)
            
            # If no specific intermediates are requested, default to all steps
            if save_intermediates_at is None:
                save_intermediates_at = list(range(len(timesteps)))
            
            # Add the initial noise as the first intermediate (before any sampling)
            if 0 in save_intermediates_at:
                intermediates.append(x_t.clone())
            
            for i in tqdm(range(len(timesteps) - 1), desc=f"DDIM Sampling with η={eta}"):
                # Current and next timestep in the subsequence
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                
                # Create a batch of timesteps
                timestep_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
                
                # Predict the noise
                predicted_noise = model(x_t, timestep_tensor)
                
                # Predict x0 from xt and the predicted noise
                predicted_x0 = self.predict_x0_from_xt(x_t, timestep_tensor, predicted_noise)
                
                # Get the current alpha values
                alpha_bar_t = self.alpha_bar[t]
                alpha_bar_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(device)
                
                # Reshape for broadcasting - use expand instead of view for scalar tensors
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).expand(n_samples, 1, 1, 1)
                
                # Calculate parameters for reverse process
                beta_t = self.beta_schedule[t]
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                
                # Calculate sigma_t based on eta
                sigma_t = eta * torch.sqrt(beta_tilde)
                
                # Calculate direction coefficient
                # sqrt(1-α̅t-1 - σ²_t) for the noise component
                direction_coef = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2).expand(n_samples, 1, 1, 1)
                
                # Add random noise if eta > 0
                random_noise = torch.randn_like(x_t) if eta > 0 and t_prev > 0 else torch.zeros_like(x_t)
                
                # DDIM update rule
                x_t_prev = sqrt_alpha_bar_prev * predicted_x0 + direction_coef * predicted_noise + sigma_t.expand(n_samples, 1, 1, 1) * random_noise
                
                # Update x_t for next iteration
                x_t = x_t_prev
                
                # Save intermediate result if this is a requested index
                if i+1 in save_intermediates_at:
                    intermediates.append(x_t.clone())

        model.train()
        return x_t, intermediates
    
    def encode_ddim(self, model, x0, device, timestep_subset=None):
        """
        Encode a real image x0 into its latent representation xT using DDIM.
        Args:
            model: The trained noise prediction model
            x0: The real image to encode
            device: Device to run the encoding on
            timestep_subset: Subset of timesteps to use for accelerated encoding
                            If None, use all timesteps
        Returns:
            The latent representation xT
        """
        model.eval()
        with torch.no_grad():
            # Start from the real image
            x_t = x0.to(device)
            
            # Define timesteps to use (forward direction)
            if timestep_subset is None:
                timesteps = list(range(0, self.timesteps))
            else:
                timesteps = sorted(timestep_subset)
            
            for i in tqdm(range(len(timesteps) - 1), desc="DDIM Encoding"):
                # Current and next timestep in the subsequence
                t_current = timesteps[i]
                t_next = timesteps[i + 1]
                
                # Create the batch of current timesteps
                timestep_tensor = torch.full((x0.shape[0],), t_current, device=device, dtype=torch.long)
                
                # Predict the noise at the current timestep
                predicted_noise = model(x_t, timestep_tensor)
                
                # Predict x0 from the current xt
                predicted_x0 = self.predict_x0_from_xt(x_t, timestep_tensor, predicted_noise)
                
                # Get the alpha values
                alpha_bar_current = self.alpha_bar[t_current]
                alpha_bar_next = self.alpha_bar[t_next]
                
                # Reshape for broadcasting
                sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next).view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_next = torch.sqrt(1 - alpha_bar_next).view(-1, 1, 1, 1)
                
                # DDIM encoding equation (deterministic forward process)
                x_t_next = sqrt_alpha_bar_next * predicted_x0 + sqrt_one_minus_alpha_bar_next * predicted_noise
                
                # Update x_t for next iteration
                x_t = x_t_next
            
        model.train()
        return x_t
