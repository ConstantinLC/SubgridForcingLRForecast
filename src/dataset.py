# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from itertools import product

class PyQGXArrayDataset(Dataset):
    def __init__(
        self,
        data_dir:str, 
        start_run: int = 0,
        end_run: int = 100, #index of final run (exclusive)
        lead_time: int = 1,
        transforms: torch.nn.Module = None, output_transforms: torch.nn.Module = None, 
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.lead_time = lead_time
        self.start_run = start_run
        self.end_run = end_run 
        self.transforms = transforms
        self.output_transforms = output_transforms
        
        run_paths_lowres = [os.path.join(self.data_dir, "lowres", f"run_{run}.nc") for run in range(self.start_run, self.end_run)] 
        print(run_paths_lowres)
        self.data_lowres = xr.open_mfdataset(run_paths_lowres, engine="h5netcdf").load()
        print('a')
        self.samples_per_run = len(self.data_lowres.time) - self.lead_time
        
        self.n_runs = self.end_run - self.start_run

        
    def __len__(self):
        return self.samples_per_run * self.n_runs


    def __getitem__(self, idx):
        run_id = idx // self.samples_per_run
        sample_id = idx % self.samples_per_run
 
        input_lowres_q = torch.from_numpy(np.array(self.data_lowres.isel(run=run_id, time=sample_id).q)).to(torch.float32)
        target_lowres_q = torch.from_numpy(np.array(self.data_lowres.isel(run=run_id, time=sample_id + self.lead_time).q)).to(torch.float32)
        input_lowres_forcing = torch.from_numpy(np.array(self.data_lowres.isel(run=run_id, time=sample_id).q_forcing)).to(torch.float32)
        
        return input_lowres_q, target_lowres_q, input_lowres_forcing

        #torch.cuda.empty_cache()