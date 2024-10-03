"""
Wasserstein GAN.

Both Generator and Discrminator are the same as DCGAN.

For more details, see: http://arxiv.org/abs/1701.07875
"""

from torch import nn
from torch import Tensor

class Generator(nn.Module):
    """Generator in WGAN."""
    def __init__(self, num_channels: int, latent_dim: int) -> None:
        """Initialize Generator.

        Args:
            num_channels(int): number of channels.
            latent_dim(int): dimensionality of the latent space.
        """
        super(Generator, self).__init__()        
        self.fc = nn.Linear(latent_dim, 1024 * 2 * 2)
        self.convs = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, num_channels, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Tanh())

    
    def forward(self, z: Tensor) -> Tensor:
        """Forward pass in Generator.
        
        Args:
            z(Tensor): latent variables of shape (N, D).
        
        Returns:
            Tensor: fake images of shape (N, C, H, W) created by Generator.
        """
        out: Tensor = self.fc(z)
        out = out.view(-1, 1024, 2, 2)
        return self.convs(out)


class Discriminator(nn.Module):
    """Discriminator in WGAN."""
    def __init__(self, num_channels: int) -> None:
        """Initialize Discriminator.
        
        Args:
            num_channels(int): number of channels.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 128, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(1024), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(1024 * 2 * 2, 1), nn.Sigmoid())
    
    def forward(self, images: Tensor) -> Tensor:
        """Forward pass in Discriminator.
        
        Args:
            images(Tensor): images, of shape (N, C, H, W)
        
        Returns:
            Tensor: probabilities of images to be real data, of shape (N, 1)
        """
        return self.model(images)
