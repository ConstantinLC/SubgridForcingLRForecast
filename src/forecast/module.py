# Taken from PDEArena
import torch
from torch import nn
import pytorch_lightning as pl
from utils.activations import ACTIVATION_REGISTRY
import torch.optim as optim
from parametrization.subgrid_parametrization import SubgridParametrization
import glob
#from kornia.filters import box_blur, blur_pool2d
from forecast.unet import UNet2d, UNet2d_hr_encoder, UNet2d_hr_encoder_bottleneck, UNet2d_hr_encoder_simple, UNet2d_hr_encoder_simple_lr
import numpy as np
#import xesmf as xe
from forecast.fno import FNO2d
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.fft as fft
from matplotlib import pyplot as plt
from torchmetrics.regression import PearsonCorrCoef

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
        pretrained_parameterization_path=None,
        pretrained_forecast_path=None,
        noise_level=0,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.add_forcing = add_forcing
        self.add_highres_encoding = add_highres_encoding
        self.add_parametrization = add_parametrization
        self.learning_rate = learning_rate
        self.highres_forecasting = highres_forecasting
        self.pretrained_parameterization_path = pretrained_parameterization_path
        self.pretrained_forecast_path = pretrained_forecast_path
        self.autoregressive = autoregressive
        self.residual_mode = residual_mode
            
        self.n_added_input_components = self.n_input_scalar_components * (int(self.add_forcing) + 
                                           int(self.add_highres_encoding) + 
                                           int(self.add_parametrization))
        
        if self.add_highres_encoding:
            self.pre_forecast_hr = UNet2d_hr_encoder(in_channels=2, out_channels=2, init_features=32)
        else:
            self.pre_forecast_lr = UNet2d(in_channels=2, out_channels=2) #UNet2d(in_channels=2, out_channels=2, init_features=32)

        if self.add_parametrization:
            self.subgrid_parametrization = SubgridParametrization(
                        n_input_scalar_components=self.n_input_scalar_components,
                        n_output_scalar_components=2)
        
        self.unet = UNet2d(in_channels=self.n_input_scalar_components, out_channels=2)


        self.noise_level = noise_level
        
        # Pretrained with laplace noise 0.2 : model-100-val_MSE0.016.ckpt, corresponding pretrained parameterization: model-163-val_MSE0.34.ckpt
        # Forecast then fine-tuned with parameterization and forecast model: model-08-val_MSE0.036.ckpt
        """if self.pretrained_forecast_path != '':
            #self.pretrained_forecast_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-08-val_MSE0.036.ckpt'
            pretrained_dict = torch.load(self.pretrained_forecast_path)['state_dict']
            self.unet.load_state_dict({k.replace("unet.",""): v for k, v in pretrained_dict.items() if "unet" in k})
            #for param in self.unet.parameters():
            #    param.requires_grad = False
            if self.add_highres_encoding:
                self.pre_forecast_hr.load_state_dict({k.replace("pre_forecast_hr.",""): v for k, v in pretrained_dict.items() if "pre_forecast_hr" in k})
            elif not self.add_parametrization:
                self.pre_forecast_lr.load_state_dict({k.replace("pre_forecast_lr.",""): v for k, v in pretrained_dict.items() if "pre_forecast_lr" in k})
            #self.inverse_subgrid_parameterization.load_state_dict({k.replace("inverse_subgrid_parameterization.",""): v for k, v in pretrained_dict.items() if "inverse_subgrid_parameterization" in k}) 
        
        if self.pretrained_parameterization_path != '':
            #self.pretrained_parameterization_path = '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-08-val_MSE0.036.ckpt'
            pretrained_dict = torch.load(self.pretrained_parameterization_path)['state_dict']

            self.subgrid_parametrization.load_state_dict({k.replace("subgrid_parametrization.",""): v for k, v in pretrained_dict.items() if "subgrid_parametrization" in k})  
            
            #for param in self.subgrid_parametrization.parameters():
            #    param.requires_grad = False
        """
    def forward(self, x, forcing=None, highres_x=None, noise=False):
        
        #distribution = torch.distributions.laplace.Laplace(loc=0, scale=0.3)
        #bn = nn.InstanceNorm2d(num_features=2, affine=False, momentum=0).to(device='cuda')
        #bn_enc = bn(hr_encoding[0])
        #print(torch.mean(bn_enc), torch.mean(hr_encoding[0]), torch.std(bn_enc))
        #lr_pred_from_hr_enc = self.inverse_subgrid_parameterization(x_pre_forecast_hr)
        #noisy_pre_forecast_hr = add_noise_at_frequency_batch(x_pre_forecast_hr, frequency_range=[0.1,0.9], noise_level=100)
        #print(torch.mean(x_pre_forecast_hr), torch.std(x_pre_forecast_hr))
        #  torch.normal(0, 0.5, size=hr_encoding.shape).to(device='cuda')
        #x_noisy = self.subgrid_parametrization(x) #hr_encoding + distribution.sample(sample_shape=hr_encoding.shape).to(device='cuda') 
        #print(x_noisy[10])
        #print(hr_encoding[10])
        #fig, axes = plt.subplots(1, 2)
        #axes[0].matshow(x_noisy[10, 0].detach().cpu())
        #axes[1].matshow(hr_encoding[10, 0].detach().cpu())
        #plt.show()
        #x_noisy = 0.1*hr_encoding + 0.9*x_noisy
        #print(nn.MSELoss()(self.pre_forecast_hr(highres_x), hr_encoding))

        #nvprint(nn.MSELoss()(hr_encoding, x_noisy))
        #plt.hist(np.ravel((x_noisy - hr_encoding).detach().cpu()))
        #plt.hist(np.ravel((self.subgrid_parametrization(x) - hr_encoding).detach().cpu()))
        #plt.show()
        #hr_encoding = x_noisy

        if self.add_highres_encoding:
            pre_forecast_encoding = self.pre_forecast_hr(highres_x)
        elif self.add_parametrization:
            pre_forecast_encoding = self.subgrid_parametrization(x)
        else:
            pre_forecast_encoding = self.pre_forecast_lr(x)

        if noise and not self.noise_level==0:
            distribution = torch.distributions.laplace.Laplace(loc=0, scale=self.noise_level)
            pre_forecast_encoding = pre_forecast_encoding + distribution.sample(sample_shape=pre_forecast_encoding.shape).to(device='cuda')
        else:
            pre_forecast_encoding = pre_forecast_encoding

        #preds = self.unet(torch.cat((x, pre_forecast_encoding), dim=1))
        preds = self.unet(x)

        return None, None, preds, None, None

    def normalize(self, tensor):
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True))/torch.std(tensor, axis=(0, 2, 3), keepdims=True)
    
    def training_step(self, batch, batch_idx):
        
        if not self.autoregressive:
            
            input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

            input_lowres_q = self.normalize(input_lowres_q)
            target_lowres_q = self.normalize(target_lowres_q)
            input_highres_q = self.normalize(input_highres_q)
            
            input = input_lowres_q
            target = target_lowres_q
            
            x_pre_forecast_hr, x_pre_forecast_lr, preds, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q, noise=True) 
                
            #loss_pred_lr_from_encoding = nn.MSELoss()(lr_pred_from_hr_enc, input_lowres_q)
            loss_forecast_hr_encoding = nn.MSELoss()(preds, target_lowres_q)
            #loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)

            loss = loss_forecast_hr_encoding #+ loss_pred_lr_from_encoding #loss_forecast_parameterization

        else: 
            
            lowres_q, lowres_forcing, highres_q = batch

            input_lowres_q = self.normalize(lowres_q[:, 0])
            target = self.normalize(lowres_q[:, -1])
            #print(input_lowres_q.shape)
            preds = input_lowres_q
            loss = None
            for i in range(lowres_q.shape[1]-1):
                #print(i)
                if self.add_highres_encoding:
                    highres_q_i = self.normalize(highres_q[:, i])
                    _, _, preds, _, _ = self.forward(preds, highres_x = highres_q_i) #+ preds
                else:
                    _, _, preds, _, _ = self.forward(preds) #+ preds

                target = self.normalize(lowres_q[:, i+1])
                if loss is None:
                    loss = nn.MSELoss()(preds, target)
                else:
                    loss += nn.MSELoss()(preds, target)

        self.log('train/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        if not self.autoregressive:

            input_lowres_q, target_lowres_q, input_lowres_forcing, input_highres_q, target_highres_q = batch

            input_lowres_q = self.normalize(input_lowres_q)
            target_lowres_q = self.normalize(target_lowres_q)
            input_highres_q = self.normalize(input_highres_q)
            
            input = input_lowres_q
            target = target_lowres_q
            
            x_pre_forecast_hr, x_pre_forecast_lr, preds, preds_lr, lr_pred_from_hr_enc = self.forward(input, highres_x = input_highres_q, noise=False) 
                
            #loss_pred_lr_from_encoding = nn.MSELoss()(lr_pred_from_hr_enc, input_lowres_q)
            loss_forecast_hr_encoding = nn.MSELoss()(preds, target_lowres_q)
            #loss_forecast_parameterization = nn.MSELoss()(preds_lr, target_lowres_q)
            print(loss_forecast_hr_encoding)

            loss = loss_forecast_hr_encoding #+ loss_pred_lr_from_encoding #loss_forecast_parameterization

        else:

            lowres_q, lowres_forcing, highres_q = batch

            input_lowres_q = self.normalize(lowres_q[:, 0])
            target = self.normalize(lowres_q[:, -1])
            #print(input_lowres_q.shape)
            preds = input_lowres_q
            for i in range(lowres_q.shape[1]-1):
                #print(i)
                if self.add_highres_encoding:
                    highres_q_i = self.normalize(highres_q[:, i])
                    _, _, preds, _, _ = self.forward(preds, highres_x = highres_q_i) #+ preds
                else:
                    _, _, preds, _, _ = self.forward(preds) #+ preds

            loss = nn.MSELoss()(preds, target)
            #print(preds.shape, target.shape)

        self.log('val/pearson_cor', PearsonCorrCoef(num_outputs=preds.shape[0]).to(device='cuda')(torch.flatten(preds, start_dim=1).T, torch.flatten(target, start_dim=1).T).mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        #lr_value = self.trainer.optimizers[0].param_groups[0]['lr']
        #self.log('learning_rate', lr_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
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