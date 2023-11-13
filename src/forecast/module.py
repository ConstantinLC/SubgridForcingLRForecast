# Taken from PDEArena
import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
import torch.optim as optim
from parametrization.subgrid_parametrization import SubgridParametrization
import glob
from kornia.filters import box_blur
from unet import UNet2d

class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if norm:
            # Original used BatchNorm2d
            self.norm1 = nn.GroupNorm(num_groups, out_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.activation(self.norm2(self.conv2(h)))
        return h


class Down(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        h = self.pool(x)
        h = self.conv(h)
        return h


class Up(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_groups=1, norm: bool = True, activation="gelu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, num_groups, norm, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.up(x1)
        h = torch.cat([x2, h], dim=1)
        h = self.conv(h)
        return h

class ForecastModel(pl.LightningModule):
    """Our interpretation of the original U-Net architecture.

    Uses [torch.nn.GroupNorm][] instead of [torch.nn.BatchNorm2d][]. Also there is no `BottleNeck` block.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use. One of ["gelu", "relu", "silu"].
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_output_scalar_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation="gelu",
        learning_rate=1e-3,
        add_forcing=False,
        add_highres_encoding=False,
        add_parametrization=False,
        highres_forecasting=False,
        parametrization_path=None
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        self.add_forcing = add_forcing
        self.add_highres_encoding = add_highres_encoding
        self.add_parametrization = add_parametrization
        self.learning_rate = learning_rate
        self.highres_forecasting = highres_forecasting
        self.parametrization_path = parametrization_path
            
        self.n_added_input_components = self.n_input_scalar_components * (int(self.add_forcing) + 
                                           int(self.add_highres_encoding) + 
                                           int(self.add_parametrization))
        
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        insize = time_history * (self.n_input_scalar_components + self.n_added_input_components)
        n_channels = insize
        #self.image_proj = ConvBlock(insize, n_channels, activation=activation)

        self.down = nn.ModuleList(
            [
                Down(n_channels, n_channels * 2, activation=activation),
                Down(n_channels * 2, n_channels * 4, activation=activation),
                Down(n_channels * 4, n_channels * 8, activation=activation),
                Down(n_channels * 8, n_channels * 16, activation=activation),
            ]
        )
        self.up = nn.ModuleList(
            [
                Up(n_channels * 16, n_channels * 8, activation=activation),
                Up(n_channels * 8, n_channels * 4, activation=activation),
                Up(n_channels * 4, n_channels * 2, activation=activation),
                Up(n_channels * 2, n_channels, activation=activation),
            ]
        )
        out_channels = time_future * (self.n_output_scalar_components)
        # should there be a final norm too? but we aren't doing "prenorm" in the original
        self.final = nn.Conv2d(n_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        #nn.Conv2d(2, 2, kernel_size=6, stride=4, padding=1),  # 64x128x40
               #nn.LeakyReLU(0.3),
        if self.add_highres_encoding:
            self.hr_encoder = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                #nn.Dropout(p=0.1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                #nn.Dropout(p=0.1),
                nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=2),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                #nn.Dropout(p=0.1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                #nn.Dropout(p=0.1),
                nn.Conv2d(32, 2, kernel_size=6, stride=2, padding=2),
            )

        self.lr_encoder = nn.Sequential(
                nn.Conv2d(self.n_input_scalar_components * (1 + int(self.add_forcing) + int(self.add_highres_encoding) + int(self.add_parametrization)), 16, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Dropout(p=0.1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Dropout(p=0.1),
                nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=2),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Dropout(p=0.1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64x128x40
                nn.LeakyReLU(0.3),
                nn.Dropout(p=0.1),
                nn.Conv2d(32, 2, kernel_size=6, stride=2, padding=2),
            )

        if self.add_parametrization:
            self.subgrid_parametrization = SubgridParametrization(
                            n_input_scalar_components=self.n_input_scalar_components,
                            n_output_scalar_components=self.n_output_scalar_components)
            
            pretrained_forecast_path = self.parametrization_path
            #'/mnt/SSD2/constantin/subgrid_modelling/checkpoints/parametrization/model-195-val_MSE0.08.ckpt'
            #for parametrization of learned encoding 
            pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
            
            pretrained_dict = {k.replace("parametrization.",""): v for k, v in pretrained_dict.items() if "parametrization" in k}
            self.subgrid_parametrization.load_state_dict(pretrained_dict)

            for param in self.subgrid_parametrization.parameters():
                param.requires_grad = False

        self.unet = UNet2d(in_channels=self.n_input_scalar_components * (1 + int(self.add_forcing) + int(self.add_highres_encoding) + int(self.add_parametrization)), out_channels=2)


    def forward(self, x, highres_x=None):

        if self.add_highres_encoding:
            x_encoded = self.hr_encoder(highres_x)
            x = torch.cat((x, x_encoded), dim=1)
        
        elif self.add_parametrization:
            x_encoded = self.subgrid_parametrization(x)
            x_cat = torch.cat((x, torch.zeros(x_encoded.shape).to(device='cuda')), dim=1)
            x_cat[:, -x_encoded.shape[1]:] = x_encoded
            x = x_cat

        #x = torch.cat((x, self.lr_encoder(x.repeat_interleave(4,dim=-1).repeat_interleave(4,dim=-2))), dim=1)

        #h = x

        #x1 = self.down[0](x)
        #x2 = self.down[1](x1)
        #x3 = self.down[2](x2)
        #x4 = self.down[3](x3)
        #x = self.up[0](x4, x3)
        #x = self.up[1](x, x2)
        #x = self.up[2](x, x1)
        #x = self.up[3](x, h)

        #x = self.final(x)

        #preds = x 

        preds = self.unet(x)
        return preds
    

    def evaluate(self, x, y, metrics, input_highres_x=None):
        
        preds = self.forward(x, input_highres_x)

        return [m(preds, y) for m in metrics]
    
    
    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)

    def training_step(self, batch, batch_idx):
        
        if self.add_highres_encoding or self.highres_forecasting:
            input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch
            input_highres_q = self.normalize(input_highres_q)
            target_highres_q = self.normalize(target_highres_q)
        else: 
            input_lowres_q, target_lowres_q, input_lowres_forcing = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        input_lowres_forcing = self.normalize(input_lowres_forcing)
        
        if self.highres_forecasting:
            input = input_highres_q
            target = target_highres_q
        else: 
            target = target_lowres_q
            if self.add_forcing:
                input = torch.cat((input_lowres_q, input_lowres_forcing), axis=1)
            else:
                input = input_lowres_q

        if self.add_highres_encoding:
            preds = self.forward(input, input_highres_q)
        else:
            preds = self.forward(input)

        loss = nn.MSELoss()(preds, target)

        self.log(
            'train/MSE',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.add_highres_encoding or self.highres_forecasting:
            input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch
            input_highres_q = self.normalize(input_highres_q)
            target_highres_q = self.normalize(target_highres_q)
        else: 
            input_lowres_q, target_lowres_q, input_lowres_forcing = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        input_lowres_forcing = self.normalize(input_lowres_forcing)

        if self.highres_forecasting:
            input = input_highres_q
            target = target_highres_q
        else: 
            target = target_lowres_q
            if self.add_forcing:
                input = torch.cat((input_lowres_q, input_lowres_forcing), axis=1)
            else:
                input = input_lowres_q

        if self.add_highres_encoding:
            preds = self.forward(input, input_highres_q)
        else:
            preds = self.forward(input)

        loss = nn.MSELoss()(preds, target)

        self.log(
            "val/" + "MSE",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer