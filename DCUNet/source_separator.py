import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio_contrib as audio_nn
import numpy as np
from .constant import *
from .utils import realimag, istft, cut_padding
from .unet import UNet


class SourceSeparator(nn.Module):
    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode):
        """
        :param complex: Whether to use complex networks.
        :param model_complexity:
        :param model_depth: Only two options are available : 10, 20
        :param log_amp: Whether to use log amplitude to estimate signals
        :param padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
        """
        super().__init__()
        self.net = nn.Sequential(
            STFT(complex=complex, log_amp=log_amp),
            UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode),
            ApplyMask(complex=complex, log_amp=log_amp),
            ISTFT(complex=complex, log_amp=log_amp)
        )

    def forward(self, x, istft=True):
        if istft:
            return self.net(x)
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
            return x

    def inference_one_audio(self, audio, normalize=True):
        """
        :param audio: channel x samples (tensor, float) 
        :return: 
        """
        audict = SourceSeparator.preprocess_audio(audio, sequence_length=16384)
        with torch.no_grad():
            for k, v in audict.items():
                audict[k] = v.unsqueeze(1).cuda()
            Y_hat = self.forward(audict, istft=False).squeeze(1)
            y_hat = istft(Y_hat, HOP_LENGTH, length=audio.shape[-1])
            if normalize:
                mx = y_hat.max(dim=-1)[0].view(y_hat.shape[0], -1)
                mn = y_hat.min(dim=-1)[0].view(y_hat.shape[0], -1)
                y_hat = 2 * (y_hat - mn) / (mx - mn) - 1.
        return y_hat
    
    @staticmethod
    def preprocess_audio(x, sequence_length=None):
        assert sequence_length is not None
        audio_length = x.shape[-1]
    
        if sequence_length is not None:
            if audio_length % sequence_length > 0:
                target_length = (audio_length // sequence_length + 1) * sequence_length
            else:
                target_length = audio_length
    
            x = cut_padding(x, target_length, np.random.RandomState(0), deterministic=True)
    
        x_max = x.max(dim=-1)[0].view(x.shape[0], -1)
        x_min = x.min(dim=-1)[0].view(x.shape[0], -1)
        x = 2 * (x - x_min) / (x_max - x_min) - 1.
    
        rt = dict(x=x,
                  x_max=x_max,
                  x_min=x_min)
    
        return rt


class STFT(nn.Module):
    def __init__(self, complex=True, log_amp=False):
        super(self.__class__, self).__init__()
        self.stft = audio_nn.STFT(fft_length=N_FFT, hop_length=HOP_LENGTH)
        self.amp2db = audio_nn.AmplitudeToDb()

        self.complex = complex
        self.log_amp = log_amp
        window = torch.hann_window(N_FFT)
        self.register_buffer('window', window)

    def forward(self, bd):
        with torch.no_grad():
            bd['X'] = self.stft(bd['x'])

            if not self.complex:
                bd['mag_X'], bd['phase_X'] = audio_nn.magphase(bd['X'], power=1.)
            if self.log_amp:
                bd['X'] = self.amp2db(bd['X'])
        return bd


class ApplyMask(nn.Module):
    def __init__(self, complex=True, log_amp=False):
        super().__init__()
        self.amp2db = audio_nn.DbToAmplitude()
        self.complex = complex
        self.log_amp = log_amp

    def forward(self, bd):
        if not self.complex:
            Y_hat = bd['mag_X'] * bd['M_hat']
            Y_hat = realimag(Y_hat, bd['phase_X'])
            if self.log_amp:
                raise NotImplementedError
        else:
            Y_hat = bd['X'] * bd['M_hat']
            if self.log_amp:
                Y_hat = self.amp2db(Y_hat)

        return Y_hat


class ISTFT(nn.Module):
    def __init__(self, complex=True, log_amp=False, length=16384):
        super().__init__()
        self.amp2db = audio_nn.DbToAmplitude()
        self.complex = complex
        self.log_amp = log_amp
        self.length = length

    def forward(self, Y_hat):
        # Y_hat : batch x channel x freq x time x 2
        num_batch = Y_hat.shape[0]
        num_channel = Y_hat.shape[1]
        Y_hat = Y_hat.view(Y_hat.shape[0] * Y_hat.shape[1], Y_hat.shape[2], Y_hat.shape[3], Y_hat.shape[4])
        y_hat = istft(Y_hat, hop_length=HOP_LENGTH, win_length=N_FFT, length=self.length)  # expected target signal
        y_hat = y_hat.view(num_batch, num_channel, -1)

        return y_hat

