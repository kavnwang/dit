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

checkpoint_epoch = 101

checkpoint = torch.load(f'checkpoints/dit_epoch_{checkpoint_epoch}.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

def display_data(x: torch.Tensor, nrows: int, title: str):
    """Saves a batch of data to file using matplotlib."""
    os.makedirs('outputs', exist_ok=True)
    
    x = x.detach().cpu()
    ncols = x.shape[0] // nrows

    # Rearrange to create a grid of images, keeping RGB channels
    y = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows)
    
    # Normalize to [0, 1] range
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    
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
latent = torch.randn((bs, 3, dit_config['img_size'], dit_config['img_size'])).to(device)
num_steps = 1000
class_label = 30
num_classes = dit_config['num_classes']
w = 15.0

timesteps = torch.linspace(dit_config['T'], 0, num_steps + 1, dtype=torch.long).to(device)

for i in range(num_steps):
    t = timesteps[i]
    t_prev = timesteps[i + 1]
    
    timestep = t.unsqueeze(0).expand(bs)
    classes = torch.tensor(class_label).unsqueeze(0).to(device).expand(bs)
    unconditional = torch.tensor(num_classes).unsqueeze(0).to(device).expand(bs)

    noise_c = model(latent, classes, timestep)
    noise_u = model(latent, unconditional, timestep)
    noise_pred = noise_u + w * (noise_c - noise_u)
    
    alpha_t = alpha_bar[t]
    alpha_t_prev = alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0).to(device)
    
    x_0_pred = (latent - (1 - alpha_t) ** 0.5 * noise_pred) / (alpha_t ** 0.5)
    
    if t_prev > 0:
        dir_xt = ((1 - alpha_t_prev) ** 0.5) * noise_pred
        latent = (alpha_t_prev ** 0.5) * x_0_pred + dir_xt
    else:
        latent = x_0_pred

display_data(latent, int(math.sqrt(bs)), "Generated Image")