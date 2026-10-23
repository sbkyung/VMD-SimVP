from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


def read_array(path, key):
    path = Path(path)
    packed = path.with_suffix(".npz")
    if packed.is_file():
        with np.load(packed, allow_pickle=False) as data:
            return data[key]
    return loadmat(str(path), variable_names=[key])[key]


@dataclass
class ForecastData:
    inputs: np.ndarray
    ecmwf: np.ndarray
    data_range: float


def load_data(args):
    shape = (args.ny, args.nx, args.n_steps)
    inputs = np.zeros(shape + (args.channel,), dtype=np.float32)

    def put(channel, path, key):
        values = read_array(path, key)
        if values.shape != shape:
            raise ValueError(f"{path}: expected {shape}, got {values.shape}")
        inputs[..., channel] = np.nan_to_num(values, nan=0, posinf=0, neginf=0)

    put(0 if args.vmd_mode == 0 else 11, args.usdm_file, "usdm")
    if args.vmd_mode:
        for mode, channel in ((1, 0), (2, 8), (3, 9), (4, 10)):
            put(channel, args.imf_template.format(mode=mode), "stacks")
        target_channel = {1: 0, 2: 8, 3: 9, 4: 10}[args.vmd_mode]
        inputs[..., [0, target_channel]] = inputs[..., [target_channel, 0]]

    for channel, filename in enumerate(args.nldas_files, 1):
        put(channel, filename, "stacks1")
    for channel, weeks in enumerate((2, 4, 8, 12), 4):
        put(channel, args.eddi_template.format(weeks=weeks), "eddi0")

    forecast = np.zeros((args.n_steps, 3, args.ny, args.nx, 2), dtype=np.float32)
    for lead in range(1, 4):
        for channel, variable in enumerate(("vmax", "tmax")):
            filename = args.ecmwf_template.format(lead=lead, variable=variable)
            values = read_array(filename, f"lead{lead}")
            if values.shape != (args.n_steps, args.ny, args.nx):
                raise ValueError(f"{filename}: unexpected shape {values.shape}")
            forecast[:, lead - 1, :, :, channel] = np.nan_to_num(
                values, nan=0, posinf=0, neginf=0)

    training_target = inputs[:, :, 312:991, 0]
    data_range = float(training_target.max() - training_target.min())
    return ForecastData(inputs, forecast, data_range)


class DroughtDataset(Dataset):
    def __init__(self, data, split, stride=1):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.data = data
        left, right = (312, 991) if split == "train" else (979, data.inputs.shape[2])
        self.starts = list(range(left, right - 24 + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        frames = self.data.inputs[:, :, start:start + 24, :].transpose(2, 3, 0, 1)
        forecast = self.data.ecmwf[start + 11].transpose(0, 3, 1, 2)
        return {
            "input": torch.from_numpy(frames[:12].copy()),
            "label": torch.from_numpy(frames[12:, :1].copy()),
            "ecmwf": torch.from_numpy(forecast.copy()),
            "sample_id": index + 1,
            "target_start_index": start + 12,
        }


def make_loader(data, args, split):
    dataset = DroughtDataset(data, split, args.stride)
    if not len(dataset):
        raise ValueError(f"No {split} samples")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                      num_workers=args.num_worker, drop_last=False)
