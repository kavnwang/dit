import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from vae import  Encoder, Decoder
from gan import Discriminator
from torchvision import transforms
from plotly.express import imshow
import json
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
    ]
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_dataset = ImageFolder('./data/tiny-imagenet-200/train', transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

vae_config = json.load(open('vae_config.json'))

encoder = Encoder(hidden_dim=vae_config['hidden_dim'], latent_dim=vae_config['latent_dim'], img_size=vae_config['img_size']).to(device)
decoder = Decoder(hidden_dim=vae_config['hidden_dim'], latent_dim=vae_config['latent_dim'], img_size=vae_config['img_size']).to(device)
loss_mse = nn.MSELoss()

job = json.load(open('job_config.json'))

def display_data(x: torch.Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    x = x.detach().cpu()
    ncols = x.shape[0] // nrows

    # Rearrange to create a grid of images, keeping RGB channels
    y = rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows)
    
    # Normalize to [0, 1] range
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    y = (y * 255).to(dtype=torch.uint8)
    
    fig = imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 4),
        width=50 * (ncols + 5),
        title=f"{title}<br>single input shape = {x[0].shape}",
    )
    fig.show()


#VAE Training
discriminator = Discriminator(img_size=vae_config['img_size'])
vae_optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=job['lr'], weight_decay=job['weight_decay'], betas=job['betas'])

gan_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=job['lr'], weight_decay=job['weight_decay'], betas=job['betas'])

for epoch in range(job['epochs']):
    for iters in range(job['vae_iters']):
        for step, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            mu, logvar = encoder(imgs)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            output = decoder(z)
            mse_loss = loss_mse(output, imgs)
            kl_div_loss = (
                0.5 * (mu**2 + logvar.exp() - logvar - 1)
            ).mean()
            loss = mse_loss + kl_div_loss * job['kl_weight']
            loss.backward()
            vae_optimizer.step()
            vae_optimizer.zero_grad()
            if step == len(train_loader) - 1:
                print(f"Loss: {loss.item()}")
                print(f"Sigma: {torch.norm(std)}")
                display_data(output, 4, "Output")

    for iters in range(job['gan_iters']):
        for step, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            real_output = discriminator(imgs)
            z = encoder(imgs)
            output = decoder(z)
            fake_output = discriminator(output)
            real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
            fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
            loss = real_loss + fake_loss
            loss.backward()
            gan_optimizer.step()
            gan_optimizer.zero_grad()
            if step == len(train_loader) - 1:
                print(f"Loss: {loss.item()}")
                display_data(output, 4, "Output")