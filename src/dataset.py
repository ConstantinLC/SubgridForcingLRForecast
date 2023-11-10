# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr
from itertools import product

# Modified version of PyQGXArrayDataset for faster high resolution data reading. For now doesn't support shuffling
class PyQGXArrayDataset_Lowres(IterableDataset):
    def __init__(
        self,
        data_dir: str, 
        start_run: int = 0,
        end_run: int = 100,  # index of the final run (exclusive)
        shuffle: bool = False,
        n_runs_per_chunk: int = 1,
        lead_time: int = 1,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.start_run = start_run
        self.end_run = end_run
        self.n_runs = self.end_run - self.start_run
        self.n_runs_per_chunk = n_runs_per_chunk
        self.shuffle = shuffle
        self.lead_time = lead_time

        self.runs_range = list(range(self.start_run, self.end_run))
        if self.shuffle:
            random.shuffle(self.runs_range)

        self.run_paths_lowres = [os.path.join(self.data_dir, "lowres", f"run_{run}.nc") for run in self.runs_range]

        self.data_lowres = xr.open_mfdataset(self.run_paths_lowres, engine="h5netcdf")

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        assert(len(self.runs_range) % num_workers==0)
        runs_per_worker = int(len(self.runs_range) // num_workers)
            
        for run_idx in self.runs_range[runs_per_worker * worker_id: runs_per_worker * (worker_id + 1)]:

            input_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            target_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            input_lowres_forcing = self.data_lowres.sel(run=run_idx).q_forcing.load()

            n_samples_per_chunk = len(self.data_lowres.time) - self.lead_time
            
            for sample_idx in range(n_samples_per_chunk):

                input_lowres_q_sample = torch.from_numpy(np.array(
                    input_lowres_q.isel(time=sample_idx))).to(torch.float32)

                target_lowres_q_sample = torch.from_numpy(np.array(
                    target_lowres_q.isel(time=sample_idx+self.lead_time))).to(torch.float32)

                input_lowres_forcing_sample = torch.from_numpy(np.array(
                    input_lowres_forcing.isel(time=sample_idx))).to(torch.float32)

                yield input_lowres_q_sample, target_lowres_q_sample, input_lowres_forcing_sample