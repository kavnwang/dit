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
#dit_config = json.load(open('dit_config.json'))

job = json.load(open('job_config.json'))


transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

train_dataset = ImageFolder('./data/tiny-imagenet-200/train', transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=job['batch_size'],
    shuffle=True
)


model = DiT().to(device)
loss_mse = nn.MSELoss()

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

optimizer = torch.optim.AdamW(model.parameters(), lr=job['lr'], weight_decay=job['weight_decay'], betas=job['betas'])

s = torch.linspace(0,1,4000+1)
eps = 1e-3
alpha_bar = torch.cos(((s + eps) / (1 + eps)) * math.pi / 2) ** 2 
alpha_bar = alpha_bar / alpha_bar[0]
alpha_bar = alpha_bar.to(device)

def noise(z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = alpha_bar[t][:,None,None,None].to(z.device)
    noise = torch.randn_like(z).to(z.device)
    noised_z = alpha * z + (1 - alpha) ** 0.5 * noise
    return noise, noised_z



for epoch in range(job['epochs']):
    for step, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        t = torch.randint(0, 4000, (imgs.shape[0],)).to(device)
        error, noised_z = noise(imgs, t)
        noise_pred, _ = model(noised_z, labels, t)
        loss = loss_mse(noise_pred, error)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Loss: {loss.item()}")
            output = (noised_z[:16] - noise_pred[:16] * (1 - alpha_bar[t][:16,None,None,None]) ** 0.5 ) / alpha_bar[t][:16,None,None,None]
            display_data(output, 4, f"Epoch {epoch+1} Output")

        if step == len(train_loader) - 1:
            print(f"Epoch {epoch+1}/{job['epochs']}")
