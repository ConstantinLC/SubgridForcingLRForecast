# Taken from PDEArena
import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
import torch.optim as optim
from parametrization.subgrid_parametrization import SubgridParametrization
import glob
from kornia.filters import box_blur, blur_pool2d
from forecast.unet import UNet2d, UNet2d_hr_encoder, UNet2d_hr_encoder_bottleneck, UNet2d_hr_encoder_simple, UNet2d_hr_encoder_simple_lr
import numpy as np
import xesmf as xe
from forecast.fno import FNO2d

class ForecastModel(pl.LightningModule):

    def __init__(
        self,
        n_input_scalar_components: int,
        n_output_scalar_components: int,
        learning_rate=1e-3,
        add_forcing=False,
        add_highres_encoding=False,
        add_parametrization=False,
        highres_forecasting=False,
        autoregressive=False,
        residual_mode=False,
        parametrization_path=None
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.add_forcing = add_forcing
        self.add_highres_encoding = add_highres_encoding
        self.add_parametrization = add_parametrization
        self.learning_rate = learning_rate
        self.highres_forecasting = highres_forecasting
        self.parametrization_path = parametrization_path
        self.autoregressive = autoregressive
        self.residual_mode = residual_mode
            
        self.n_added_input_components = self.n_input_scalar_components * (int(self.add_forcing) + 
                                           int(self.add_highres_encoding) + 
                                           int(self.add_parametrization))
        
        if self.add_highres_encoding:
            self.pre_forecast_hr = UNet2d_hr_encoder_simple(in_channels=2, out_channels=2) #UNet2d_hr_encoder_bottleneck(in_channels=2, out_channels=2, init_features=16)
        elif not self.add_parametrization:
            self.pre_forecast_lr = UNet2d(in_channels=2, out_channels=2)

        self.unet = UNet2d(in_channels=self.n_input_scalar_components * 2, out_channels=2)
        
        if self.add_parametrization:
            
            self.subgrid_parametrization = SubgridParametrization(
                            n_input_scalar_components=self.n_input_scalar_components,
                            n_output_scalar_components=2)
            
            pretrained_forecast_path = self.parametrization_path
            #for parametrization of learned encoding  
            pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
            
            pretrained_dict = {k.replace("parametrization.",""): v for k, v in pretrained_dict.items() if "parametrization" in k}
            self.subgrid_parametrization.load_state_dict(pretrained_dict)

            for param in self.subgrid_parametrization.parameters():
                param.requires_grad = False

            """ pretrained_forecast_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-36-val_MSE0.022.ckpt' #model-37-val_MSE0.035.ckpt' #model-26-val_MSE0.03.ckpt' #model-38-val_MSE0.03.ckpt' #model-199-val_MSE0.01-v1.ckpt'
                #for parametrization of learned encoding  
                pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
                pretrained_dict = {k.replace("unet.",""): v for k, v in pretrained_dict.items() if "unet" in k}
                self.unet.load_state_dict(pretrained_dict)
                for param in self.unet.parameters():
                    param.requires_grad = False

                self.subgrid_parametrization = UNet2d_hr_encoder_simple_lr(in_channels=self.n_input_scalar_components, out_channels=2)"""
            
                

    def forward(self, x, forcing=None, highres_x=None):
        if self.add_highres_encoding:
            x_pre_forecast = self.pre_forecast_hr(highres_x)
        elif self.add_forcing:
            x_pre_forecast = self.pre_forecast_lr(forcing)
        elif not self.add_parametrization: #if not self.highres_forecasting:
            x_pre_forecast = self.pre_forecast_lr(x)

        if self.add_parametrization:
            x_encoded = self.subgrid_parametrization(x)
            #x_cat = torch.cat((x, torch.zeros(x_encoded.shape).to(device='cuda')), dim=1)
            #x_cat[:, -x_encoded.shape[1]:] = x_encoded
            x_pre_forecast = x_encoded
            #x = x_cat

        x = torch.cat((x, x_pre_forecast), dim=1)

        preds = self.unet(x)
        return preds
    
    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)

    def training_step(self, batch, batch_idx):

        if not self.autoregressive:
        
            if self.add_highres_encoding or self.highres_forecasting:
                input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch
                input_highres_q = self.normalize(input_highres_q)
                target_highres_q = self.normalize(target_highres_q)
            else: 
                input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

            input_lowres_q = self.normalize(input_lowres_q)
            target_lowres_q = self.normalize(target_lowres_q)
            input_lowres_forcing = self.normalize(input_lowres_forcing)
            
            if self.highres_forecasting:
                input = input_highres_q
                target = target_highres_q
            else: 
                input = input_lowres_q
                target = target_lowres_q
                
            if self.add_highres_encoding:
                preds = self.forward(input, highres_x = input_highres_q) 
            elif self.add_forcing:
                preds = self.forward(input, forcing = input_lowres_forcing)
            else:
                preds = self.forward(input)

            loss = nn.MSELoss()(preds, target)

        else:

            lowres_q, lowres_forcing, highres_q = batch
        
            input_lowres_q = self.normalize(lowres_q[:, 0])

            preds = input_lowres_q
            loss = None
            for i in range(lowres_q.shape[1]-1):

                if self.add_highres_encoding:
                    highres_q_i = self.normalize(highres_q[:, i])
                    preds = self.forward(preds, highres_x = highres_q_i) + preds
                else:
                    preds = self.forward(preds) + preds

                target = self.normalize(lowres_q[:, i+1])
                if loss is None:
                    loss = nn.MSELoss()(preds, target)
                else:
                    loss += nn.MSELoss()(preds, target)

        self.log(
            'train/MSE',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False, 
            sync_dist=True,)

        return loss

    def validation_step(self, batch, batch_idx):

        if not self.autoregressive:

            if self.add_highres_encoding or self.highres_forecasting:
                input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch
                input_highres_q = self.normalize(input_highres_q)
                target_highres_q = self.normalize(target_highres_q)
            else: 
                input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch = batch

            input_lowres_q = self.normalize(input_lowres_q)
            target_lowres_q = self.normalize(target_lowres_q)
            input_lowres_forcing = self.normalize(input_lowres_forcing)
            
            if self.highres_forecasting:
                input = input_highres_q
                target = target_highres_q
            else: 
                input = input_lowres_q
                target = target_lowres_q
                
            if self.add_highres_encoding:
                preds = self.forward(input, highres_x = input_highres_q)
            elif self.add_forcing:
                preds = self.forward(input, forcing = input_lowres_forcing)
            else:
                preds = self.forward(input)
    
        else:

            lowres_q, lowres_forcing, highres_q = batch
        
            input_lowres_q = self.normalize(lowres_q[:, 0])
            target = self.normalize(lowres_q[:, -1])

            preds = input_lowres_q
            for i in range(lowres_q.shape[1]-1):

                if self.add_highres_encoding:
                    highres_q_i = self.normalize(highres_q[:, i])
                    preds = self.forward(preds, highres_x = highres_q_i) + preds
                else:
                    preds = self.forward(preds) + preds

                #preds += self.normalize(lowres_q[:, i])

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


class ForecastModelJoint(pl.LightningModule):

    def __init__(
        self,
        n_input_scalar_components: int,
        n_output_scalar_components: int,
        learning_rate=1e-3,
        add_forcing=False,
        add_highres_encoding=False,
        add_parametrization=False,
        highres_forecasting=False,
        autoregressive=False,
        residual_mode=False,
        parametrization_path=None
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.add_forcing = add_forcing
        self.add_highres_encoding = add_highres_encoding
        self.add_parametrization = add_parametrization
        self.learning_rate = learning_rate
        self.highres_forecasting = highres_forecasting
        self.parametrization_path = parametrization_path
        self.autoregressive = autoregressive
        self.residual_mode = residual_mode
            
        self.n_added_input_components = self.n_input_scalar_components * (int(self.add_forcing) + 
                                           int(self.add_highres_encoding) + 
                                           int(self.add_parametrization))
        
        self.pre_forecast_hr = UNet2d_hr_encoder_simple(in_channels=2, out_channels=2, init_features=32)
        
        self.unet = UNet2d(in_channels=self.n_input_scalar_components * 2, out_channels=2)
        
        self.subgrid_parametrization = SubgridParametrization(
                        n_input_scalar_components=self.n_input_scalar_components,
                        n_output_scalar_components=2)

    def forward(self, x, forcing=None, highres_x=None, noise=False):
        
        x_pre_forecast_hr = self.pre_forecast_hr(highres_x)
        x_pre_forecast_lr = self.subgrid_parametrization(x)
        
        if noise:
            preds_hr = self.unet(torch.cat((x, x_pre_forecast_hr + torch.randn(x_pre_forecast_hr.shape).to(device='cuda') * 0.1), dim=1))
        else:
            preds_hr = self.unet(torch.cat((x, x_pre_forecast_hr), dim=1))
            #x_pre_forecast_hr += torch.randn(x_pre_forecast_hr.shape).to(device='cuda') * 0.01
        
        
        preds_lr = self.unet(torch.cat((x, x_pre_forecast_lr), dim=1))

        return x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr

    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)

    def training_step(self, batch, batch_idx):
        
        input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        input_lowres_forcing = self.normalize(input_lowres_forcing)
        
        input = input_lowres_q
        target = target_lowres_q
        
        x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr = self.forward(input, highres_x = input_highres_q, noise=True) 
            
        #loss_joint = nn.MSELoss()(x_pre_forecast_hr, x_pre_forecast_lr)
        loss_joint = nn.MSELoss()(preds_hr, preds_lr)
        loss_forecast_hr_encoding = nn.MSELoss()(preds_hr, target_lowres_q)
        loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)

        loss = loss_joint + loss_forecast_hr_encoding + loss_forecast_parameterization

        self.log('train/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/MSE_joint', loss_joint, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/MSE_forecast_hr_encoding', loss_forecast_hr_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/MSE_forecast_parameterization', loss_forecast_parameterization, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        
        input = input_lowres_q
        target = target_lowres_q
        
        x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr = self.forward(input, highres_x = input_highres_q) 
            
        loss_joint = nn.MSELoss()(x_pre_forecast_hr, x_pre_forecast_lr)
        loss_forecast_hr_encoding = nn.MSELoss()(preds_hr, target_lowres_q)
        loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)

        loss = 10*loss_joint + loss_forecast_hr_encoding + loss_forecast_parameterization

        self.log('val/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/MSE_joint', loss_joint, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/MSE_forecast_hr_encoding', loss_forecast_hr_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/MSE_forecast_parameterization', loss_forecast_parameterization, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer