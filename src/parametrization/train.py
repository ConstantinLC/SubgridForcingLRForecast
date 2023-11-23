import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from module import ParametrizationTeachingTrueForcing, ParametrizationTeachingLearnedForcing
from datamodule import DataModule
from forecast.module import ForecastModel

from utils.utils_config import parse_arguments, load_configuration

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')

def main():
    # Define your data module

    args = parse_arguments()
    config_file = args.config
    config = load_configuration(config_file)

    wandb.init(
        # set the wandb project where this run will be logged
        project="parametrization_training",
        name=config_file,
        config=config
    )
    
    highres = config['pipeline']['add_highres_encoding']

    datamodule = DataModule(data_dir="/mnt/SSD2/constantin/data/pyqg", batch_size=128, 
                            num_workers=5, highres=highres, shuffle=not highres)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/MSE',
        mode='min',
        dirpath='/mnt/SSD2/constantin/subgrid_modelling/checkpoints/parametrization',
        filename='model-{epoch:02d}-val_MSE{val/MSE:.2f}',
        auto_insert_metric_name=False
    )

    if config['pipeline']['add_forcing']:
        parametrization_teaching = ParametrizationTeachingTrueForcing(
            n_input_scalar_components=2,
            n_output_scalar_components=2)
    else:
        parametrization_teaching = ParametrizationTeachingLearnedForcing(
            pretrained_forecast_path= '/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast/model-75-val_MSE0.06.ckpt',
            n_input_scalar_components=2,
            n_output_scalar_components=2)
    parametrization_teaching.cuda()

    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=config['training']['max_epochs'], accelerator="gpu", devices=1, callbacks=[checkpoint_callback], logger=wandb_logger)#, overfit_batches=1)

    trainer.fit(parametrization_teaching, datamodule)

if __name__ == '__main__':
    main()