import os

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr
import random

import threading

# Modified version of PyQGXArrayDataset for faster high resolution data reading. For now doesn't support shuffling
class PyQGXArrayDataset_Highres(IterableDataset):
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
        self.run_paths_highres = [os.path.join(self.data_dir, "highres", f"run_{run}.nc") for run in self.runs_range]

        self.data_lowres = xr.open_mfdataset(self.run_paths_lowres, engine="h5netcdf")
        self.data_highres = xr.open_mfdataset(self.run_paths_highres, engine="h5netcdf")

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        assert(len(self.runs_range) % num_workers==0)
        runs_per_worker = int(len(self.runs_range) // num_workers)
            
        for run_idx in self.runs_range[runs_per_worker * worker_id: runs_per_worker * (worker_id + 1)]:

            """input_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            target_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            input_lowres_forcing = self.data_lowres.sel(run=run_idx).q_forcing.load()
            """
            input_highres_q = self.data_highres.sel(run=run_idx).q.load()
            target_highres_q = self.data_highres.sel(run=run_idx).q.load()

            input_lowres_q = input_highres_q[..., ::4, ::4]
            target_lowres_q = target_highres_q[..., ::4, ::4]
            input_lowres_forcing = self.data_lowres.sel(run=run_idx).q_forcing.load()

            n_samples_per_chunk = len(self.data_lowres.time) - self.lead_time
            
            indices = list(range(n_samples_per_chunk))
            if self.shuffle:
                random.shuffle(indices)

            for sample_idx in indices:

                input_lowres_q_sample = torch.from_numpy(np.array(
                    input_lowres_q.isel(time=sample_idx))).to(torch.float32)

                target_lowres_q_sample = torch.from_numpy(np.array(
                    target_lowres_q.isel(time=sample_idx+self.lead_time))).to(torch.float32)

                input_lowres_forcing_sample = torch.from_numpy(np.array(
                    input_lowres_forcing.isel(time=sample_idx))).to(torch.float32)

                input_highres_q_sample = torch.from_numpy(np.array( 
                    input_highres_q.isel(time=sample_idx))).to(torch.float32)

                target_highres_q_sample = torch.from_numpy(np.array(
                    target_highres_q.isel(time=sample_idx+self.lead_time))).to(torch.float32)

                yield input_lowres_q_sample, target_lowres_q_sample, input_lowres_forcing_sample, input_highres_q_sample, target_highres_q_sample



class PyQGXArrayDataset_Highres_AR(IterableDataset):
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
        self.run_paths_highres = [os.path.join(self.data_dir, "highres", f"run_{run}.nc") for run in self.runs_range]

        self.data_lowres = xr.open_mfdataset(self.run_paths_lowres, engine="h5netcdf")
        self.data_highres = xr.open_mfdataset(self.run_paths_highres, engine="h5netcdf")

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        assert(len(self.runs_range) % num_workers==0)
        runs_per_worker = int(len(self.runs_range) // num_workers)
            
        for run_idx in self.runs_range[runs_per_worker * worker_id: runs_per_worker * (worker_id + 1)]:

            """input_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            target_lowres_q = self.data_lowres.sel(run=run_idx).q.load()
            input_lowres_forcing = self.data_lowres.sel(run=run_idx).q_forcing.load()
            """
            input_highres_q = self.data_highres.sel(run=run_idx).q.load()
            target_highres_q = self.data_highres.sel(run=run_idx).q.load()

            input_lowres_q = input_highres_q[..., ::4, ::4]
            target_lowres_q = target_highres_q[..., ::4, ::4]
            input_lowres_forcing = self.data_lowres.sel(run=run_idx).q_forcing.load()

            n_samples_per_chunk = len(self.data_lowres.time) - self.lead_time
            indices = list(range(n_samples_per_chunk))

            for sample_idx in indices:

                lowres_q_sample = torch.from_numpy(np.array(
                    input_lowres_q.isel(time=range(sample_idx, sample_idx + self.lead_time)))).to(torch.float32)

                lowres_forcing_sample = torch.from_numpy(np.array(
                    input_lowres_forcing.isel(time=range(sample_idx, sample_idx + self.lead_time)))).to(torch.float32)

                highres_q_sample = torch.from_numpy(np.array( 
                    input_highres_q.isel(time=range(sample_idx, sample_idx + self.lead_time)))).to(torch.float32)

                yield lowres_q_sample, lowres_forcing_sample, highres_q_sample