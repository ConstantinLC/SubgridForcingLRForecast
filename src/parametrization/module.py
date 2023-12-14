import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
import torch.optim as optim
from forecast.module import ForecastModel, ForecastModelJoint
from subgrid_parametrization import SubgridParametrization
from kornia.filters import box_blur, blur_pool2d
    
class ParametrizationTeachingLearnedForcing(pl.LightningModule):
    def __init__(self, 
                pretrained_forecast_path: str,
                n_input_scalar_components: int = 2,
                n_output_scalar_components: int = 2,
                img_size=[64, 64]
        ):
        super(ParametrizationTeachingLearnedForcing, self).__init__()

        self.teacher_model = ForecastModelJoint(n_input_scalar_components, n_output_scalar_components,
                                        add_forcing=False, add_highres_encoding=True).pre_forecast_hr

        pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
        self.teacher_model.load_state_dict({k.replace("pre_forecast_hr.",""): v for k, v in pretrained_dict.items() if "pre_forecast_hr" in k}) 

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.parametrization = SubgridParametrization(n_input_scalar_components, n_output_scalar_components, img_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, highres_x):
        #self.teacher_model.eval()
        target = self.teacher_model(highres_x)
        output = self.parametrization(x)

        return output, target

    def training_step(self, batch, batch_idx):
        input_lowres_q, _, _, input_highres_q, _ = batch
        input_highres_q = self.normalize(input_highres_q)
        input_lowres_q = self.normalize(input_lowres_q)
        
        output, target = self.forward(input_lowres_q, input_highres_q)

        loss = self.loss_fn(output, target)
        self.log(
            "train/" + "MSE",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_lowres_q, _, _, input_highres_q, _ = batch
        input_highres_q = self.normalize(input_highres_q)
        input_lowres_q = self.normalize(input_lowres_q)
        
        output, target = self.forward(input_lowres_q, input_highres_q)
        loss = self.loss_fn(output, target)
        self.log(
            "val/" + "MSE",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss
    
    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    

class ParametrizationTeachingTrueForcing(pl.LightningModule):
    def __init__(self, 
                n_input_scalar_components: int = 2,
                n_output_scalar_components: int = 2,
                img_size=[64, 64]
        ):
        super(ParametrizationTeachingTrueForcing, self).__init__()

        self.parametrization = SubgridParametrization(n_input_scalar_components, n_output_scalar_components, img_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        output = self.parametrization(x)

        return output

    def training_step(self, batch, batch_idx):
        input_lowres_q, _, input_forcing_q = batch
        input_forcing_q = self.normalize(input_forcing_q)
        input_lowres_q = self.normalize(input_lowres_q)
        
        output = self.forward(input_lowres_q)

        loss = self.loss_fn(output, input_forcing_q)
        self.log(
            "train/" + "MSE",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_lowres_q, _, input_forcing_q = batch
        input_forcing_q = self.normalize(input_forcing_q)
        input_lowres_q = self.normalize(input_lowres_q)
        
        output = self.forward(input_lowres_q)
        loss = self.loss_fn(output, input_forcing_q)
        self.log(
            "val/" + "MSE",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss
    
    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
