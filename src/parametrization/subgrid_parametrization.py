import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY

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
        self.model = nn.Sequential(
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
        )

    def forward(self, x):
        preds = self.model(x)
        return preds