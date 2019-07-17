import torch
import numpy as np
import soundfile
import librosa
import torch.nn.functional as F


def cut_padding(y, required_length, random_state, deterministic=False):

    if isinstance(y, list):
        audio_length = y[0].shape[-1]
    else:
        audio_length = y.shape[-1]

    if audio_length < required_length:
        if deterministic:
            pad_left = 0
        else:
            pad_left = random_state.randint(required_length - audio_length + 1)  # 0 ~ 50 random
        pad_right = required_length - audio_length - pad_left  # 50~ 0

        if isinstance(y, list):
            for i in range(len(y)):
                y[i] = F.pad(y[i], (pad_left, pad_right))
            audio_length = y[0].shape[-1]
        else:
            y = F.pad(y, (pad_left, pad_right))
            audio_length = y.shape[-1]

    if deterministic:
        audio_begin = 0
    else:
        audio_begin = random_state.randint(audio_length - required_length + 1)
    audio_end = required_length + audio_begin
    if isinstance(y, list):
        for i in range(len(y)):
            y[i] = y[i][..., audio_begin:audio_end]
    else:
        y = y[..., audio_begin:audio_end]
    return y


def load_audio(path, sample_rate, assert_sr=False, channel=None):
    if path[-3:] == "pcm":
        audio, sr = soundfile.read(path, format="RAW", samplerate=16000, channels=1, subtype="PCM_16",
                                   dtype="float32")
    else:
        audio, sr = soundfile.read(path, dtype="float32")

    if len(audio.shape) == 1:  # if mono
        audio = np.expand_dims(audio, 1)

    # samples x channel
    # sr이 16000이 아니면 resample
    if assert_sr:
        assert sr == sample_rate

    if sr != sample_rate:
        audio = librosa.core.resample(audio.T, sr, sample_rate).T

    # assert sr == SAMPLE_RATE
    audio = torch.FloatTensor(audio).permute(1, 0)
    if channel is not None:
        audio = audio[:channel]

    return dict(audio=audio, path=path)  # channel x samples

def get_audio_by_magphase(mag, phase, hop_length, n_fft, length=None):
    # mag : channel x freq x time
    # phase : channel x freq x time
    mono_audio_stft = realimag(mag, phase)
    # channel x freq x time x 2

    mono_audio = istft(mono_audio_stft, hop_length, n_fft, length=length)
    return mono_audio


def _get_time_values(sig_length, sr, hop):
    """
    Get the time axis values given the signal length, sample
    rate and hop size.
    """
    return torch.linspace(0, sig_length/sr, sig_length//hop+1)


def _get_freq_values(n_fft, sr):
    """
    Get the frequency axis values given the number of FFT bins
    and sample rate.
    """
    return torch.linspace(0, sr/2, n_fft//2 + 1)


def get_spectrogram_axis(sig_length, sr, n_fft=2048, hop=512):
    t = _get_time_values(sig_length, sr, hop)
    f = _get_freq_values(n_fft, sr)
    return t, f


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    # keunwoochoi's implementation
    # https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e

    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == "hann"
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = F.pad(y, (0, length - y.shape[1]))
            # y = torch.cat((y[:, :length], torch.zeros(y.shape[0], length - y.shape[1])))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


def angle(tensor):
    """
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(tensor[...,1], tensor[...,0])


def magphase(spec, power=1.):
    """
    Separate a complex-valued spectrogram with shape (*,2)
    into its magnitude and phase.
    """
    mag = spec.pow(2).sum(-1).pow(power/2)
    phase = angle(spec)
    return mag, phase


def realimag(mag, phase):
    """
    Combine a magnitude spectrogram and a phase spectrogram to a complex-valued spectrogram with shape (*, 2)
    """
    spec_real = mag * torch.cos(phase)
    spec_imag = mag * torch.sin(phase)
    spec = torch.stack([spec_real, spec_imag], dim=-1)
    return spec


def get_snr(y, z):
    y_power = y.pow(2).mean(dim=-1)
    z_power = z.pow(2).mean(dim=-1)
    snr = 20*torch.log10(y_power/z_power)
    return snr
