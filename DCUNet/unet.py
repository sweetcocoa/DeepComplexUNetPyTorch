import torch
import torch.nn as nn
import torch.nn.functional as F
import DCUNet.complex_nn as complex_nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels=1,
                 complex=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros"):
        super().__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=complex)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)

        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, bd):
        if self.complex:
            x = bd['X']
        else:
            x = bd['mag_X']
        # go down
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            #print("x{}".format(i), x.shape)
            x = encoder(x)
        # xs : x0=input x1 ... x9

        #print(x.shape)
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)

        #print(p.shape)
        mask = self.linear(p)
        mask = torch.tanh(mask)
        bd['M_hat'] = mask
        return bd

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 ]
            self.enc_kernel_sizes = [(7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            self.enc_strides = [(2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 1)]
            self.enc_paddings = [(2, 1),
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 4),
                                     (6, 4),
                                     (6, 4),
                                     (7, 5)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2)]

            self.dec_paddings = [(1, 1),
                                 (1, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1)]

        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (6, 3),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (2, 1),
                                 (2, 1),
                                 (0, 3),
                                 (3, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
