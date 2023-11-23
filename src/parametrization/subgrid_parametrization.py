import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
from forecast.unet import UNet2d
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
        )      """      


        self.model = UNet2d(self.n_input_scalar_components, self.n_output_scalar_components)
        
        """self.model = nn.Sequential(
            nn.Conv2d(self.n_input_scalar_components, 256, kernel_size=5, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, self.n_output_scalar_components, kernel_size=3, padding='same'),  # 64x128x40
            nn.LeakyReLU(0.2)
        )"""

    def forward(self, x):
        preds = self.model(x)
        return preds
'''
class UNet2d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )'''