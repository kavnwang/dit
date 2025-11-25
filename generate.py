import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from dit import DiT
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import math
from typing import Tuple
import os
import sys


dit_config = json.load(open('dit_config.json'))
job = json.load(open('job_config.json'))

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = DiT(
    num_blocks=dit_config['num_blocks'],
    hidden_dim=dit_config['hidden_dim'], 
    heads=dit_config['heads'], 
    img_size=dit_config['img_size'], 
    patches=dit_config['patches'], 
    T=dit_config['T'], 
    eps=dit_config['eps'], 
    num_classes=dit_config['num_classes'], 
    embedding_dim=dit_config['embedding_dim'], 
    condition_p=dit_config['condition_p'], 
    intermediate_ratio=dit_config['intermediate_ratio'], 
    adaln_intermediate_ratio=dit_config['adaln_intermediate_ratio']
).to(device)

checkpoint_epoch = 281

checkpoint = torch.load(f'checkpoints/dit_epoch_{checkpoint_epoch}.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

def display_data(x: torch.Tensor, nrows: int, title: str):
    """Saves a batch of data to file using matplotlib."""
    os.makedirs('outputs', exist_ok=True)
    
    x = x.detach().cpu()
    x = (x * 0.5) + 0.5
    x = torch.clamp(x, 0, 1)
    
    ncols = x.shape[0] // nrows

    # Rearrange to create a grid of images, keeping RGB channels
    y = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows)
    
    # Save using matplotlib
    fig, ax = plt.subplots(figsize=(ncols * 2, nrows * 2))
    ax.imshow(y.numpy())
    ax.axis('off')
    ax.set_title(f"{title}\nSingle input shape = {x[0].shape}", fontsize=10)
    plt.tight_layout()
    
    # Create safe filename from title
    safe_title = title.replace(' ', '_').replace('/', '_')
    filename = f"outputs/{safe_title}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved image to {filename}")


s = torch.linspace(0,1,4000+1)
eps = dit_config['eps']
alpha_bar = torch.cos(((s + eps) / (1 + eps)) * math.pi / 2) ** 2 
alpha_bar = alpha_bar / alpha_bar[0]
alpha_bar = alpha_bar.to(device)

bs = 1
num_steps = 1000
class_label = 1
num_classes = dit_config['num_classes']
w = 1.0

timesteps = torch.linspace(dit_config['T'], 0, num_steps + 1, dtype=torch.long).to(device)

transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


dataset = ImageFolder('./data/tiny-imagenet-200/train', transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

def add_noise(x: torch.Tensor, t: int) -> torch.Tensor:
    """Add noise to an image at timestep t using forward diffusion."""
    noise = torch.randn_like(x)
    alpha_t = alpha_bar[t]
    noised = (alpha_t ** 0.5) * x + ((1 - alpha_t) ** 0.5) * noise
    return noised

def generate(latent: torch.Tensor, classes: torch.Tensor, start_timestep: int = None) -> torch.Tensor:
    """Denoise a latent image iteratively from start_timestep to 0 using DDPM sampling."""
    with torch.no_grad():
        # Create custom timestep schedule based on start_timestep
        if start_timestep is not None:
            # Always denoise for num_steps steps starting from start_timestep
            custom_timesteps = torch.linspace(start_timestep, 0, num_steps + 1, dtype=torch.long).to(device)
        else:
            # Use default timesteps
            custom_timesteps = timesteps
            
        for i in range(num_steps):
            t = custom_timesteps[i]
            t_prev = custom_timesteps[i + 1]
            
            timestep = t.unsqueeze(0).expand(bs)
            unconditional = torch.tensor(num_classes).unsqueeze(0).to(device).expand(bs)

            noise_c = model(latent, classes, timestep)
            noise_u = model(latent, unconditional, timestep)
            noise_pred = noise_u + w * (noise_c - noise_u)
            
            alpha_bar_t = alpha_bar[t]
            alpha_bar_t_prev = alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0).to(device)
            
            # Calculate alpha_t (not alpha_bar)
            alpha_t = alpha_bar_t / alpha_bar_t_prev
            beta_t = 1 - alpha_t
            
            if t_prev > 0:
                # DDPM: Calculate posterior mean using reparameterized formula
                mean = (1.0 / (alpha_t ** 0.5)) * (latent - (beta_t / ((1 - alpha_bar_t) ** 0.5)) * noise_pred)
                
                # DDPM: Calculate posterior variance
                posterior_variance = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
                
                # Add random noise (stochastic sampling)
                noise = torch.randn_like(latent)
                latent = mean + (posterior_variance ** 0.5) * noise
            else:
                # Final step: predict x_0 directly
                latent = (latent - ((1 - alpha_bar_t) ** 0.5) * noise_pred) / (alpha_bar_t ** 0.5)
    return latent

# Choose how many timesteps to noise forward
noise_timestep = 3000  # You can adjust this (0 to T, where higher = more noise)

for step, (image, label) in enumerate(tqdm(dataloader)):
    if step >= 10:  # Process only first 10 images, adjust as needed
        break
        
    image = image.to(device)
    label = label.to(device)
    
    # Display original image
    display_data(image, 1, f"Step_{step}_1_Original")
    
    # Noise the image forward to noise_timestep
    noised_image = add_noise(image, noise_timestep)
    display_data(noised_image, 1, f"Step_{step}_2_Noised_t{noise_timestep}")
    
    # Denoise the image back to 0
    denoised_image = generate(noised_image, label, start_timestep=noise_timestep)
    display_data(denoised_image, 1, f"Step_{step}_3_Denoised")