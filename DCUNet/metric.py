import torch
import torch.functional as F

from .constant import *
from .utils import istft, realimag
from pypesq import pesq


class PESQ:
    def __init__(self):
        self.pesq = pesq_metric

    def __call__(self, output, bd):
        return self.pesq(output, bd)


def pesq_metric(y_hat, bd):
    # PESQ
    with torch.no_grad():
        y_hat = y_hat.cpu().numpy()
        y = bd['y'].cpu().numpy()  # target signal

        sum = 0
        for i in range(len(y)):
            sum += pesq(y[i, 0], y_hat[i, 0], SAMPLE_RATE)

        sum /= len(y)
        return torch.tensor(sum)