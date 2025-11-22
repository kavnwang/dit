import torch
import torch.nn as nn
import typing

class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        img_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # 4x4x256 = 4096
        self.linear1 = nn.Linear(4 * 4 * 256, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 2 * self.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 3, 64, 64)
        y = self.relu(self.conv1(x))  # (batch, 32, 32, 32)
        y = self.relu(self.conv2(y))  # (batch, 64, 16, 16)
        y = self.relu(self.conv3(y))  # (batch, 128, 8, 8)
        y = self.relu(self.conv4(y))  # (batch, 256, 4, 4)
        
        y = self.flatten(y)  # (batch, 4096)
        y = self.relu(self.linear1(y))  # (batch, hidden_dim)
        y = self.linear2(y)  # (batch, 2 * latent_dim)
        
        mu, sigma = y[...,:self.latent_dim], y[...,self.latent_dim:]
        return mu, sigma
         
class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        img_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 4 * 4 * 256)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.linear1(x))  # (batch, hidden_dim)
        y = self.relu(self.linear2(y))  # (batch, 4096)
        
        y = torch.reshape(y, (-1, 256, 4, 4))  # (batch, 256, 4, 4)
        
        y = self.relu(self.deconv1(y))  # (batch, 128, 8, 8)
        y = self.relu(self.deconv2(y))  # (batch, 64, 16, 16)
        y = self.relu(self.deconv3(y))  # (batch, 32, 32, 32)
        y = self.sigmoid(self.deconv4(y))  # (batch, 3, 64, 64)
        
        return y
