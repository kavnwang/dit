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

vae_config = json.load(open('vae_config.json'))

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


encoder = Encoder(hidden_dim=vae_config['hidden_dim'], latent_dim=vae_config['latent_dim'], img_size=vae_config['img_size']).to(device)
decoder = Decoder(hidden_dim=vae_config['hidden_dim'], latent_dim=vae_config['latent_dim'], img_size=vae_config['img_size']).to(device)
loss_mse = nn.MSELoss()


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
        height=150 * (nrows + 2),
        width=150 * (ncols + 2),
        title=f"{title}<br>single input shape = {x[0].shape}",
    )
    fig.show()

discriminator = Discriminator(img_size=vae_config['img_size']).to(device)
loss_bce = nn.BCELoss()

# Initialize optimizers for VAE-GAN training
vae_optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=job['lr'], weight_decay=job['weight_decay'], betas=job['betas'])
disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=job['lr'], weight_decay=job['weight_decay'], betas=job['betas'])

for epoch in range(job['epochs']):
    for step, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Encode and decode
        mu, logvar = encoder(imgs)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        output = decoder(z)
        real_output = discriminator(imgs)
        fake_output = discriminator(output.detach())  # Detach to avoid backprop through generator
        real_loss = loss_bce(real_output, torch.ones_like(real_output).to(device))
        fake_loss = loss_bce(fake_output, torch.zeros_like(fake_output).to(device))
        loss_disc = real_loss + fake_loss
        loss_disc.backward()
        disc_optimizer.step()
        disc_optimizer.zero_grad()

        fake_output_gen = discriminator(output)  # Fresh forward pass without detach
        adversarial_loss = loss_bce(fake_output_gen, torch.ones_like(fake_output_gen).to(device))
        # Train VAE (encoder + decoder) with reconstruction, KL divergence, and adversarial loss
        mse_loss = loss_mse(output, imgs)
        kl_div_loss = (0.5 * (mu**2 + logvar.exp() - logvar - 1)).mean()
        
        # Combined VAE-GAN loss
        loss_vae_gan = mse_loss + kl_div_loss * job['kl_weight'] + adversarial_loss.clamp(max=2.0) * job['adversarial_weight']
        loss_vae_gan.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
        vae_optimizer.step()
        vae_optimizer.zero_grad()
        
        if step == len(train_loader) - 1:
            print(f"Epoch {epoch+1}/{job['epochs']}")
            print(f"VAE-GAN Loss: {loss_vae_gan.item():.4f} (MSE: {mse_loss.item():.4f}, KL: {kl_div_loss.item():.4f}, Adv: {adversarial_loss.item():.4f})")
            print(f"Sigma: {torch.norm(std):.4f}")
            print(f"Discriminator Loss: {loss_disc.item():.4f}")
            if epoch % 5 == 0:
                display_data(output, 4, f"Epoch {epoch+1} Output")