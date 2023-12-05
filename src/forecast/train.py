import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from datamodule import DataModule
from forecast.module import ForecastModel, ForecastModelJoint
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
        project="joint_hr_lr_training-hr_encoding_predicts_lr",
        name=config_file,
        config=config
    )
    wandb.define_metric("val/MSE", summary="min")

    add_highres_data = config['pipeline']['add_highres_encoding'] or config['pipeline']['highres_forecasting']

    datamodule = DataModule(data_dir="/mnt/SSD2/constantin/data/pyqg", batch_size=config['training']['batch_size'], 
                            num_workers=config['data']['num_workers'], highres=add_highres_data, 
                            shuffle=not config['pipeline']['add_highres_encoding'], lead_time=config['data']['lead_time'],
                            autoregressive=config['pipeline']['autoregressive'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val/MSE',
        mode='min',
        dirpath='/mnt/SSD2/constantin/subgrid_modelling/checkpoints/forecast',
        filename='model-{epoch:02d}-val_MSE{val/MSE:.3f}',
        auto_insert_metric_name=False
    )

    if config['pipeline']['add_parametrization'] and config['pipeline']['joint_training']:
        model = ForecastModelJoint(n_input_scalar_components=config['model']['n_input_scalar_components'],
                          n_output_scalar_components=config['model']['n_output_scalar_components'], 
                          learning_rate=float(config['model']['learning_rate']),
                          add_forcing=config['pipeline']['add_forcing'], 
                          add_highres_encoding=config['pipeline']['add_highres_encoding'],
                          add_parametrization=config['pipeline']['add_parametrization'],
                          highres_forecasting=config['pipeline']['highres_forecasting'],
                          parametrization_path=config['pipeline']['parametrization_path'],
                          autoregressive=config['pipeline']['autoregressive'])

    else:
        model = ForecastModel(n_input_scalar_components=config['model']['n_input_scalar_components'],
                            n_output_scalar_components=config['model']['n_output_scalar_components'], 
                            learning_rate=float(config['model']['learning_rate']),
                            add_forcing=config['pipeline']['add_forcing'], 
                            add_highres_encoding=config['pipeline']['add_highres_encoding'],
                            add_parametrization=config['pipeline']['add_parametrization'],
                            highres_forecasting=config['pipeline']['highres_forecasting'],
                            parametrization_path=config['pipeline']['parametrization_path'],
                            autoregressive=config['pipeline']['autoregressive'])


    model.cuda()

    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=config['training']['max_epochs'], accelerator="gpu", devices=1, callbacks=[checkpoint_callback], logger=wandb_logger)

    trainer.fit(model, datamodule)

if __name__ == '__main__':
    main()
