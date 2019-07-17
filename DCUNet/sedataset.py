import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
from .constant import *
from .utils import load_audio, cut_padding

"""
Note:
Target Dataset: https://datashare.is.ed.ac.uk/handle/10283/2791

psquare
bus
cafe
living
office
"""
class SEDataset(Dataset):
    def __init__(self, signals, mixtures,
                 seed=0,
                 sequence_length=16384,
                 is_validation=False,
                 preload=False,
                 ):

        super(self.__class__, self).__init__()
        self.signals = signals
        self.mixtures = mixtures
        self.is_validation = is_validation
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.preload = preload

        print("Got", len(signals), "signals and", len(mixtures), "mixtures.")

        if self.preload:
            self.data_y = []
            print("Loading Signal Data")
            for signal in tqdm(self.signals):
                self.data_y.append(load_audio(signal, SAMPLE_RATE, assert_sr=True, channel=1))

            self.data_x = []
            print("Loading Mixture Data")
            for noise in tqdm(self.mixtures):
                self.data_x.append(load_audio(noise, SAMPLE_RATE, assert_sr=True, channel=1))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.preload:
            x = self.data_x[idx]['audio']
            y = self.data_y[idx]['audio']  # channel x samples
            z = x - y
        else:
            x = load_audio(self.mixtures[idx], SAMPLE_RATE, assert_sr=True, channel=1)['audio']
            y = load_audio(self.signals[idx], SAMPLE_RATE, assert_sr=True, channel=1)['audio']
            z = x - y

        if self.sequence_length is not None:
            x, y, z = cut_padding([x, y, z], self.sequence_length, self.random, deterministic=self.is_validation)

        x_max = x.max(dim=-1)[0].view(x.shape[0], -1)
        x_min = x.min(dim=-1)[0].view(x.shape[0], -1)

        # Inverse : x = x + 1 (x + x_min  ) / 2
        x = 2 * (x - x_min) / (x_max - x_min) - 1.
        y = 2 * (y - x_min) / (x_max - x_min) - 1.
        z = 2 * (z - x_min) / (x_max - x_min) - 1.

        rt = dict(x=x,
                  y=y,
                  z=z,
                  x_max=x_max,
                  x_min=x_min)

        return rt