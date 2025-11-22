import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for GAN that classifies images as real or fake.
    """
    
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
        img_size: int = 64,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        
        self.conv1 = nn.Conv2d(num_channels, hidden_channels[0], 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], 4, 2, 1)
        
        self.norm1 = nn.BatchNorm2d(hidden_channels[0])
        self.norm2 = nn.BatchNorm2d(hidden_channels[1])
        self.norm3 = nn.BatchNorm2d(hidden_channels[2])
        
        final_size = img_size // (2 ** 3)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(final_size * final_size * hidden_channels[2], 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input images of shape (batch_size, num_channels, img_size, img_size)
            
        Returns:
            Probability that input is real, shape (batch_size, 1)
        """

        x = self.leaky_relu(self.norm1(self.conv1(x)))
        x = self.leaky_relu(self.norm2(self.conv2(x)))
        x = self.leaky_relu(self.norm3(self.conv3(x)))
        
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x