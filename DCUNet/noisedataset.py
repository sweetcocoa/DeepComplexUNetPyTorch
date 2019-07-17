import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
from .constant import *
from .utils import load_audio, cut_padding


class NoiseDataset(Dataset):
    def __init__(self, signals, noises,
                 seed=0,
                 sequence_length=16384,
                 is_validation=False,
                 snr_range=(-10, 20),
                 preload=False):

        super(self.__class__, self).__init__()
        self.signals = signals  # ['path/001.wav', 'path/002.flac', ... ]
        self.noises = noises    # ['path/a.wav', 'path/b.flac', ... ]
        self.is_validation = is_validation
        self.snr_range = snr_range
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.preload = preload  # load wav files on RAM. It reduces Disk I/O, but may consume huge memory.

        print("Got", len(signals), "signals And", len(noises), "noises.")

        if self.preload:
            self.data_y = []
            print("Loading Signal Data")
            for signal in tqdm(self.signals):
                self.data_y.append(load_audio(signal, SAMPLE_RATE, assert_sr=True, channel=1))

            self.data_z = []
            print("Loading Noise Data")
            for noise in tqdm(self.noises):
                self.data_z.append(load_audio(noise, SAMPLE_RATE, assert_sr=True, channel=1))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):

        if self.preload:
            y = self.data_y[idx]['audio']  # channel x samples
            noise_idx = self.random.randint(len(self.data_z))  # channel x samples
            z = self.data_z[noise_idx]['audio']

        else:
            y = load_audio(self.signals[idx], SAMPLE_RATE, assert_sr=True, channel=1)['audio']
            noise_idx = self.random.randint(len(self.noises))  # channel x samples
            
            # Pitch augmentation
            if not self.is_validation and self.random.uniform(0., 1.) < 0.3:
                pitch = np.random.uniform(0.6, 1.3)
                z = load_audio(self.noises[noise_idx], int(SAMPLE_RATE*pitch), assert_sr=False, channel=1)['audio']
            else:
                z = load_audio(self.noises[noise_idx], SAMPLE_RATE, assert_sr=True, channel=1)['audio']

        if self.sequence_length is not None:
            y = cut_padding(y, self.sequence_length, self.random, self.is_validation)

        power_y = y.pow(2).mean(dim=-1).squeeze(0)
        power_z = z.pow(2).mean(dim=-1).squeeze(0)
        # TODO SNR에 맞춰 볼륨조절 어딘가로 모듈화, 제대로된 validation을 위해 일정하게 SNR 합성하기
        target_SNR = self.random.randint(*self.snr_range)
        noise_factor = torch.sqrt(power_y / (power_z) / (10 ** (target_SNR / 20)))

        audio_length = y.shape[-1]
        noise_length = z.shape[-1]
        if noise_length < audio_length:
            z = cut_padding(z, audio_length, self.random, self.is_validation)
            noise_length = z.shape[-1]

        if self.is_validation:
            noise_begin = 0
        else:
            noise_begin = self.random.randint(noise_length - audio_length + 1)

        noise_end = noise_begin + audio_length
        z = z[:, noise_begin:noise_end]

        z *= noise_factor

        x = y + z

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
                  x_min=x_min,
                  power_y=power_y,
                  power_z=power_z,
                  SNR=target_SNR)
                  # signal_path=self.data_y[idx]['path'],
                  # noise_path=self.data_z[noise_idx]['path'])

        return rt
    