import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
from forecast.unet import UNet2d, UNet2d_hr_encoder_simple_lr
from collections import OrderedDict


class SubgridParametrization(nn.Module):
    def __init__(self,
        n_input_scalar_components: int = 2,
        n_output_scalar_components: int = 2,
        img_size=[64, 64],
    ):
        super(SubgridParametrization, self).__init__()

        self.img_size = img_size
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        
        # Encoder layers
        """self.model = nn.Sequential(
            nn.Conv2d(self.n_input_scalar_components, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(128, self.n_output_scalar_components, kernel_size=3, padding='same'),
        )"""
        
        self.model = nn.Sequential(
            nn.Conv2d(self.n_input_scalar_components, 256, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, self.n_output_scalar_components, kernel_size=3, padding='same')  # 64x128x40
        )

    def forward(self, x):
        preds = self.model(x)
        return preds

'''
class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=10):
        super(ResNetGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Create Residual Blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks += [ResidualBlock(64)]

        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)
'''

class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4):
        super(ResNetGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding='same'),
            nn.Tanh()
        )

        # Create Residual Blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks += [ResidualBlock(256)]

        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding='same'),
            nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        return x + self.block(x)