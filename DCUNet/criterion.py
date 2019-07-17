import torch


class WeightedSDR:
    def __init__(self):
        self.loss = weighted_signal_distortion_ratio_loss

    def __call__(self, output, bd):
        return self.loss(output, bd)


def dotproduct(y, y_hat):
    # batch x channel x nsamples
    return torch.bmm(y.view(y.shape[0], 1, y.shape[-1]), y_hat.view(y_hat.shape[0], y_hat.shape[-1], 1)).reshape(-1)


def weighted_signal_distortion_ratio_loss(output, bd):
    y = bd['y']  # target signal
    z = bd['z']  # noise signal

    y_hat = output
    z_hat = bd['x'] - y_hat  # expected noise signal

    # mono channel only...
    # can i fix this?
    y_norm = torch.norm(y, dim=-1).squeeze(1)
    z_norm = torch.norm(z, dim=-1).squeeze(1)
    y_hat_norm = torch.norm(y_hat, dim=-1).squeeze(1)
    z_hat_norm = torch.norm(z_hat, dim=-1).squeeze(1)

    def loss_sdr(a, a_hat, a_norm, a_hat_norm):
        return dotproduct(a, a_hat) / (a_norm * a_hat_norm + 1e-8)

    alpha = y_norm.pow(2) / (y_norm.pow(2) + z_norm.pow(2) + 1e-8)
    loss_wSDR = -alpha * loss_sdr(y, y_hat, y_norm, y_hat_norm) - (1 - alpha) * loss_sdr(z, z_hat, z_norm, z_hat_norm)

    return loss_wSDR.mean()
