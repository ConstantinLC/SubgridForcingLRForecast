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
            target_lowres_q = self.data_lowres.sel(run=run_idx).q.load()"""

            input_lowres_q = self.data_highres.sel(run=run_idx).q.load()[..., ::4, ::4]
            target_lowres_q = self.data_highres.sel(run=run_idx).q.load()[..., ::4, ::4]
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

                yield input_lowres_q_sample, target_lowres_q_sample, input_lowres_forcing_sample



class KolmogorovDataset_Rozet(Dataset):
    def __init__(self, name: str, folderPath: str, mode: str, resolution: str, sequenceLength:List[Tuple[int, int]]=[],
                 framesPerTimeStep: int = 1, limit_trajectories: Optional[int] = None, usegrid: bool = False, conditioned: bool = False) -> None:
        super().__init__()
        self.name = name
        self.folderPath = folderPath
        self.mode = mode
        self.resolution = resolution
        self.sequenceLength = sequenceLength
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        self.conditioned = conditioned
        
        self.seqLength = sequenceLength[0]
        self.time_step = sequenceLength[1]

        self.time_gaps = np.linspace(0, self.time_step, framesPerTimeStep, dtype = int, endpoint=False)

        file_path = os.path.join(folderPath, self.mode + ".h5")
        self.data = torch.Tensor(np.array(h5py.File(file_path, mode='r')["x"]))

        self.n_trajectories = self.data.shape[0]
        print("aaaa", self.n_trajectories)
        if self.limit_trajectories is not None:
            self.n_trajectories = min(self.n_trajectories, self.limit_trajectories)
        self.n_frames = self.data.shape[1] - self.seqLength + 1  # Ignore timestep for now
          

    def __len__(self) -> int:
        return self.n_trajectories * self.n_frames

    def __getitem__(self, idx:int) -> dict:
        idx_sim = idx // self.n_frames
        idx_frame = idx % self.n_frames
        data_idx = self.data[idx_sim][idx_frame:idx_frame+self.seqLength]
        #data_idx_lf, data_idx_hf = self.separateFrequencies(data_idx, cutoff_frequency=8)
        #data_idx = KolmogorovFlow.upsample(data_idx, 2, mode="bicubic")
        return {"data" : data_idx, "simParameters": {}} # data_idx_lf + data_idx_hf/4 # data_idx_lf*4 + data_idx_hf