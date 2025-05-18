import torch

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Define a linear variance schedule.
    
    Args:
        timesteps (int): Number of timesteps
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
        
    Returns:
        torch.Tensor: Beta schedule tensor
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine variance schedule as proposed in https://arxiv.org/abs/2102.09672
    
    Args:
        timesteps (int): Number of timesteps
        s (float): Small offset to prevent beta from being too small near t=0
        
    Returns:
        torch.Tensor: Beta schedule tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
