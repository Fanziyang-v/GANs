"""
Vanilla GAN.

Both Generator and Driscriminator are MLP.

For more details, see: https://arxiv.org/abs/1406.2661
"""

from torch import nn
from torch import Tensor

class Discriminator(nn.Module):
    def __init__(self, image_size: int) -> None:
        super(Discriminator, self).__init__()
        self.layer1 = affine_lkrelu_dropout(image_size, 1024)
        self.layer2 = affine_lkrelu_dropout(1024, 512)
        self.layer3 = affine_lkrelu_dropout(512, 256)
        self.layer4 = affine_sigmoid(256, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Generator(nn.Module):
    def __init__(self, image_size: int, latent_size: int) -> None:
        super(Generator, self).__init__()
        self.layer1 = affine_lkrelu(latent_size, 256)
        self.layer2 = affine_lkrelu(256, 512)
        self.layer3 = affine_lkrelu(512, 1024)
        self.layer4 = affine_tanh(1024, image_size)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


# affine - leaky relu - dropout
def affine_lkrelu_dropout(in_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3))


# affine - leaky relu
def affine_lkrelu(in_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2))


# affine - sigmoid
def affine_sigmoid(in_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid())


# affine - tanh
def affine_tanh(in_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Tanh())
