"""
Deep Convolutional GAN.

Both Generator and Discrminator are CNN.

For more details, see: http://arxiv.org/abs/1511.06434
"""

from torch import nn
from torch import Tensor

class Generator(nn.Module):
    def __init__(self, num_channels: int, latent_size: int) -> None:
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.layer1 = deconv_bn_relu(latent_size, 1024, kernel_size=4, stride=1, padding=0)    # 1024 x 4 x 4
        self.layer2 = deconv_bn_relu(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)            # 512 x 8 x 8
        self.layer3 = deconv_bn_relu(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)             # 256 x 16 x 16
        self.layer4 = deconv_bn_relu(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)             # 128 x 32 x 32
        self.layer5 = deconv_tanh(128, num_channels, kernel_size=5, stride=2, padding=2, output_padding=1)       # num_channels x 64 x 64

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super(Discriminator, self).__init__()
        self.layer1 = conv_leaky_relu(num_channels, 128, kernel_size=5, stride=2, padding=2)        # 128 x 32 x 32
        self.layer2 = conv_bn_leaky_relu(128, 256, kernel_size=5, stride=2, padding=2)              # 256 x 16 x 16
        self.layer3 = conv_bn_leaky_relu(256, 512, kernel_size=5, stride=2, padding=2)              # 512 x 8 x 8
        self.layer4 = conv_bn_leaky_relu(512, 1024, kernel_size=5, stride=2, padding=2)             # 1024 x 4 x 4
        self.layer5 = conv_sigmoid(1024, 1, kernel_size=4, stride=1, padding=0)                     # 1 x 1 x 1

    def forward(self, x: Tensor) -> Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

# Deconvolutional Block( Deconv - batchnorm - relu )
def deconv_bn_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int=0, output_padding: int=0) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


# Deconvolutional Block( Deconv - tanh )
def deconv_tanh(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int=0, output_padding: int=0) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.Tanh())


# Convolutional Block( Conv - leaky_relu)
def conv_leaky_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2))


# Convolutional Block( Conv - batch norm - leaky_relu)
def conv_bn_leaky_relu(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2))


# Convolutional Block ( Conv - sigmoid )
def conv_sigmoid(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Sigmoid())
