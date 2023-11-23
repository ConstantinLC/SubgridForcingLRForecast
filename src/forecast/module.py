# Taken from PDEArena
import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
import torch.optim as optim
from parametrization.subgrid_parametrization import SubgridParametrization
import glob
from kornia.filters import box_blur, blur_pool2d
from forecast.unet import UNet2d, UNet2d_hr_encoder
import numpy as np
import xesmf as xe
from fno import FNO2d

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
        
        if self.add_highres_encoding:
            self.hr_encoder = UNet2d_hr_encoder(in_channels=2, out_channels=2)
        else:
            self.hr_encoder = UNet2d(in_channels=2, out_channels=2)
        
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

        self.unet = UNet2d(in_channels=self.n_input_scalar_components * (1 + int(self.add_forcing) + 1 + int(self.add_parametrization)), out_channels=2)
        #self.unet = FNO2d(num_channels=2)

    def forward(self, x, highres_x=None):

        if self.add_highres_encoding:
            x_encoded = self.hr_encoder(box_blur(highres_x, kernel_size=(4,4))) #x.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)) #self.hr_encoder(box_blur(highres_x, kernel_size=(4,4)))
            x = torch.cat((x, x_encoded), dim=1)
        else:
            x_encoded = self.hr_encoder(x) 
            x = torch.cat((x, x_encoded), dim=1)
        
        if self.add_parametrization:
            x_encoded = self.subgrid_parametrization(x)
            x_cat = torch.cat((x, torch.zeros(x_encoded.shape).to(device='cuda')), dim=1)
            x_cat[:, -x_encoded.shape[1]:] = x_encoded
            x = x_cat            

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

        #preds = self.normalize(target_highres_q)

        #grid_in = {"lat": np.linspace(-90, 90, 128), "lon": np.linspace(-180, 180, 256)}
        #grid_out = {"lat": np.linspace(-90, 90, 32), "lon": np.linspace(-180, 180, 64)}
        #regridder = xe.Regridder(grid_in, grid_out, 'bilinear', periodic=True)
        #preds = regridder(preds)
        #preds = blur_pool2d(preds, kernel_size=4, stride=4)

        #if self.highres_forecasting:
        #    preds = blur_pool2d(preds, kernel_size=4, stride=4)

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