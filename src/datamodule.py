import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset import PyQGXArrayDataset_Lowres
from dataset_highres import PyQGXArrayDataset_Highres, PyQGXArrayDataset_Highres_AR

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, shuffle, highres = False, lead_time=1, autoregressive=False, num_workers=4):
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.highres = highres
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.autoregressive = autoregressive
        
        self.end_run_train = 80
        self.end_run_val = 100
        

    def train_dataloader(self):
        if self.autoregressive:
            train_ds = PyQGXArrayDataset_Highres_AR(data_dir=self.data_dir, start_run=0, end_run=self.end_run_train, shuffle=True, lead_time=self.lead_time)
        else:
            train_ds = PyQGXArrayDataset_Highres(data_dir=self.data_dir, start_run=0, end_run=self.end_run_train, shuffle=True, lead_time=self.lead_time)
        return DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=1)
    
    def val_dataloader(self):
        if self.autoregressive:
            val_ds = PyQGXArrayDataset_Highres_AR(data_dir=self.data_dir, start_run=self.end_run_train, end_run=self.end_run_val, shuffle=False, lead_time=self.lead_time)
        else:
            val_ds = PyQGXArrayDataset_Highres(data_dir=self.data_dir, start_run=self.end_run_train, end_run=self.end_run_val, shuffle=False, lead_time=self.lead_time)
        return DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=1)
    
    def test_dataloader(self):
        test_ds = self.dataset_type(data_dir=self.data_dir, start_run=self.end_run_val, end_run=100, lead_time=self.lead_time)
        return DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=1)
