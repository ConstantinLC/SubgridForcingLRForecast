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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.fft as fft
from matplotlib import pyplot as plt

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
            
            """pretrained_forecast_path = self.parametrization_path
            #for parametrization of learned encoding  
            pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
            
            pretrained_dict = {k.replace("parametrization.",""): v for k, v in pretrained_dict.items() if "parametrization" in k}
            self.subgrid_parametrization.load_state_dict(pretrained_dict)

            for param in self.subgrid_parametrization.parameters():
                param.requires_grad = False"""
            print('a')
            pretrained_forecast_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-187-val_MSE0.019.ckpt' #model-37-val_MSE0.035.ckpt' #model-26-val_MSE0.03.ckpt' #model-38-val_MSE0.03.ckpt' #model-199-val_MSE0.01-v1.ckpt'
            #for parametrization of learned encoding  
                #for parametrization of learned encoding  
            #for parametrization of learned encoding  
            pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
            pretrained_dict = {k.replace("unet.",""): v for k, v in pretrained_dict.items() if "unet" in k}
            self.unet.load_state_dict(pretrained_dict)
            for param in self.unet.parameters():
                param.requires_grad = False

            self.subgrid_parametrization = SubgridParametrization(
                        n_input_scalar_components=self.n_input_scalar_components,
                        n_output_scalar_components=2)

                #self.subgrid_parametrization = UNet2d_hr_encoder_simple_lr(in_channels=self.n_input_scalar_components, out_channels=2)
            
                

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
                x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q) 
                loss = nn.MSELoss()(preds_hr, target)
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
                x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q) 
                loss = nn.MSELoss()(preds_hr, target)
            elif self.add_forcing:
                preds = self.forward(input, forcing = input_lowres_forcing)
            else:
                preds = self.forward(input)
                loss = nn.MSELoss()(preds, target)
                print(loss)
    
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


def add_noise_at_frequency_batch(images, frequency_range, noise_level):
    # Convert images to PyTorch tensor
    images_tensor = torch.tensor(images, dtype=torch.float32)

    # Perform 2D FFT for each image in the batch
    fft_results = fft.fft2(images_tensor, dim=(2, 3))

    # Shift zero frequency components to the center
    fft_results_shifted = fft.fftshift(fft_results, dim=(2, 3))

    # Find the indices corresponding to the frequency range
    frequencies = torch.fft.fftfreq(images.shape[-2], 1.0).numpy()
    frequency_indices = ((frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1]))

    # Add noise to the specified frequencies for each image in the batch
    fft_results_shifted[:, :, frequency_indices, :] += noise_level
    fft_results_shifted[:, :, :, frequency_indices] += noise_level

    # Shift back to the original position
    fft_results_unshifted = fft.ifftshift(fft_results_shifted, dim=(2, 3))

    # Perform Inverse 2D FFT for each image in the batch to get back to the spatial domain
    noisy_images = fft.ifft2(fft_results_unshifted, dim=(2, 3))

    # Convert back to numpy array and get the real part
    noisy_images = noisy_images.real.to(device='cuda')

    return noisy_images


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
        
        self.pre_forecast_hr = UNet2d_hr_encoder(in_channels=2, out_channels=2, init_features=32)
        
        self.unet = UNet2d(in_channels=self.n_input_scalar_components * 2, out_channels=2)

        self.subgrid_parametrization = SubgridParametrization(
                        n_input_scalar_components=self.n_input_scalar_components,
                        n_output_scalar_components=2)
        
        self.inverse_subgrid_parameterization = SubgridParametrization(
                        n_input_scalar_components=self.n_input_scalar_components,
                        n_output_scalar_components=2) 

        """pretrained_forecast_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-198-val_MSE0.027.ckpt' #model-175-val_MSE0.073.ckpt' #

        pretrained_dict = torch.load(pretrained_forecast_path)['state_dict']
        self.unet.load_state_dict({k.replace("unet.",""): v for k, v in pretrained_dict.items() if "unet" in k})

        #for param in self.unet.parameters():
        #    param.requires_grad = False
        #self.pre_forecast_hr.load_state_dict({k.replace("pre_forecast_hr.",""): v for k, v in pretrained_dict.items() if "pre_forecast_hr" in k})
        #self.inverse_subgrid_parameterization.load_state_dict({k.replace("inverse_subgrid_parameterization.",""): v for k, v in pretrained_dict.items() if "inverse_subgrid_parameterization" in k})      
        
        pretrained_parameterization_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/parametrization/model-53-val_MSE2.14.ckpt'
        pretrained_dict = torch.load(pretrained_parameterization_path)['state_dict']

        self.subgrid_parametrization.load_state_dict({k.replace("parametrization.",""): v for k, v in pretrained_dict.items() if "parametrization" in k})  

        #for param in self.subgrid_parametrization.parameters():
        #    param.requires_grad = False"""

    def forward(self, x, forcing=None, highres_x=None, noise=False):
        
        #nv shr_encoding = self.pre_forecast_hr(highres_x)
        #distribution = torch.distributions.laplace.Laplace(loc=0, scale=0.3)
        
        #lr_pred_from_hr_enc = self.inverse_subgrid_parameterization(x_pre_forecast_hr)
        #noisy_pre_forecast_hr = add_noise_at_frequency_batch(x_pre_forecast_hr, frequency_range=[0.1,0.9], noise_level=100)
        #print(torch.mean(x_pre_forecast_hr), torch.std(x_pre_forecast_hr))
        #  torch.normal(0, 0.5, size=hr_encoding.shape).to(device='cuda')
        x_noisy = self.subgrid_parametrization(x) #hr_encoding + distribution.sample(sample_shape=hr_encoding.shape).to(device='cuda') 
        #print(x_noisy[0])
        #print(x_pre_forecast_hr[0])
        #print(nn.MSELoss()(hr_encoding, x_noisy))
        #x_noisy = 0.1*hr_encoding + 0.9*x_noisy
        #print(nn.MSELoss()(hr_encoding, x_noisy))

        #nvprint(nn.MSELoss()(hr_encoding, x_noisy))
        #plt.hist(np.ravel((x_noisy - hr_encoding).detach().cpu()))
        #plt.hist(np.ravel((self.subgrid_parametrization(x) - hr_encoding).detach().cpu()))
        #plt.show()
        hr_encoding = x_noisy
        noise=False
        if noise:
            distribution = torch.distributions.laplace.Laplace(loc=0, scale=0.2)
            x_pre_forecast_hr = hr_encoding + distribution.sample(sample_shape=hr_encoding.shape).to(device='cuda')
        else:
            x_pre_forecast_hr = hr_encoding
        
        """x_encoded = self.subgrid_parametrization(x)
        x_cat = torch.cat((x, torch.zeros(x_encoded.shape).to(device='cuda')), dim=1)
        x_cat[:, -x_encoded.shape[1]:] = x_encoded
        
        x_pre_forecast_hr = x_encoded"""

        preds_hr = self.unet(torch.cat((x, x_pre_forecast_hr), dim=1))
        #lr_pred_from_hr_encoding = self.inverse_subgrid_parameterization(hr_encoding)
        #x_pre_forecast_hr = x_pre_forecast_hr + torch.normal(0, 0.025, size=x_pre_forecast_hr.shape).to(device='cuda')

        return None, None, preds_hr, None, None

    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)
    
    def training_step(self, batch, batch_idx):
        
        input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        input_highres_q = self.normalize(input_highres_q)
        
        input = input_lowres_q
        target = target_lowres_q
        
        x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q, noise=True) 
            
        #loss_pred_lr_from_encoding = nn.MSELoss()(lr_pred_from_hr_enc, input_lowres_q)
        loss_forecast_hr_encoding = nn.MSELoss()(preds_hr, target_lowres_q)
        #loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)

        loss = loss_forecast_hr_encoding #+ loss_pred_lr_from_encoding #loss_forecast_parameterization

        self.log('train/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log('train/MSE_pred_lr_from_encoding', loss_pred_lr_from_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/MSE_forecast_hr_encoding', loss_forecast_hr_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log('train/MSE_forecast_parameterization', loss_forecast_parameterization, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

        input_lowres_q = self.normalize(input_lowres_q)
        target_lowres_q = self.normalize(target_lowres_q)
        input_highres_q = self.normalize(input_highres_q)
        
        input = input_lowres_q
        target = target_lowres_q
        
        x_pre_forecast_hr, x_pre_forecast_lr, preds_hr, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q, noise=False) 
            
        #loss_pred_lr_from_encoding = nn.MSELoss()(lr_pred_from_hr_enc, input_lowres_q)
        loss_forecast_hr_encoding = nn.MSELoss()(preds_hr, target_lowres_q)
        #loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)
        print(loss_forecast_hr_encoding)

        loss = loss_forecast_hr_encoding #+ loss_pred_lr_from_encoding #loss_forecast_parameterization

        self.log('val/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log('val/MSE_pred_lr_from_encoding', loss_pred_lr_from_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/MSE_forecast_hr_encoding', loss_forecast_hr_encoding, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log('val/MSE_forecast_parameterization', loss_forecast_parameterization, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        lr_value = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)

        T_max = 10
        T_0 = 50000
        eta_min = 1e-6

        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=eta_min)
        
        return {
            'optimizer': optimizer,
            
        }
            
        """'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/MSE',  # Optional: Monitor a metric for the scheduler
                'interval': 'epoch',  # Optional: Adjust learning rate on 'epoch' or 'step' basis
                'frequency': 1  # Optional: Adjust learning rate every 'frequency' epochs or steps
            }"""