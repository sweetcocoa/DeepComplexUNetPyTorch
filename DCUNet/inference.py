from .constant import *
from .utils import load_audio

def any_audio_inference(path, net, sequence_length=None, normalize=True):
    audio = load_audio(path, SAMPLE_RATE)['audio']
    y_hat = net.inference_one_audio(audio, normalize=normalize)
    return y_hat.cpu().numpy()
