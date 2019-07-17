import torch
import torch.nn as nn
import torch.optim as optim
import glob, os
import numpy as np

from torch.utils.data import DataLoader

from torchcontrib.optim import SWA
import PinkBlack.trainer

from DCUNet.constant import *
from DCUNet.noisedataset import NoiseDataset
from DCUNet.sedataset import SEDataset
from DCUNet.source_separator import SourceSeparator
from DCUNet.criterion import WeightedSDR
from DCUNet.metric import PESQ

args = PinkBlack.io.setup(default_args=dict(gpu="0",
                                            batch_size=12,
                                            train_signal="/data/jongho/data/2019challenge/ss/reference/16k/clean_trainset_28spk_wav/",
                                            train_noise="/data/jongho/data/2019challenge/ss/reference/16k/noisy_trainset_28spk_wav/",
                                            test_signal="/data/jongho/data/2019challenge/ss/reference/16k/clean_testset_wav/",
                                            test_noise="/data/jongho/data/2019challenge/ss/reference/16k/noisy_testset_wav/",
                                            sequence_length=16384,
                                            num_step=100000,
                                            validation_interval=500,
                                            num_workers=0,
                                            ckpt="unet/ckpt.pth",
                                            model_complexity=45,
                                            lr=0.01,
                                            num_signal=0,
                                            num_noise=0,
                                            optimizer="adam",
                                            lr_decay=0.5,
                                            momentum=0,
                                            multi_gpu=False,
                                            complex=False,
                                            model_depth=20,
                                            swa=False,
                                            loss="wsdr",
                                            log_amp=False,
                                            metric="pesq",
                                            train_dataset="se",
                                            valid_dataset="se",
                                            preload=False,          # Whether to load datasets on memory 
                                            padding_mode="reflect", # conv2d's padding mode
                               ))


def get_dataset(args):
    def get_wav(dir):
        wavs = []
        wavs.extend(glob.glob(os.path.join(dir, "**/*.wav"), recursive=True))
        wavs.extend(glob.glob(os.path.join(dir, "**/*.flac"), recursive=True))
        wavs.extend(glob.glob(os.path.join(dir, "**/*.pcm"), recursive=True))
        return wavs

    if args.train_dataset == "mix":
        train_signals = get_wav(args.train_signal)
        train_noises = get_wav(args.train_noise)
    else:
        train_signals = get_wav(args.train_signal)
        train_noises = [signal.replace("clean", "noisy") for signal in train_signals]

    if args.valid_dataset == "mix":
        test_signals = get_wav(args.test_signal)
        test_noises = get_wav(args.test_noise)
    else:
        test_signals = get_wav(args.test_signal)
        test_noises = [signal.replace("clean", "noisy") for signal in test_signals]

    if args.num_signal > 0:
        train_signals = train_signals[:args.num_signal]
        test_signals = test_signals[:args.num_signal]

    if args.num_noise > 0:
        train_noises = train_noises[:args.num_noise]
        test_noises = test_noises[:args.num_noise]

    if args.train_dataset == "mix":
        train_dset = NoiseDataset(train_signals, train_noises, sequence_length=args.sequence_length, is_validation=False, preload=args.preload)
    else:
        train_noises = [signal.replace("clean", "noisy") for signal in train_signals]
        train_dset = SEDataset(train_signals, train_noises, sequence_length=args.sequence_length, is_validation=False)

    if args.valid_dataset == "mix":
        rand = np.random.RandomState(0)
        rand.shuffle(test_signals)
        test_signals = test_signals[:1000]
        valid_dset = NoiseDataset(test_signals, test_noises, sequence_length=args.sequence_length, is_validation=True, preload=args.preload)
    else:
        test_noises = [signal.replace("clean", "noisy") for signal in test_signals]
        valid_dset = SEDataset(test_signals, test_noises, sequence_length=args.sequence_length, is_validation=True)

    return dict(train_dset=train_dset,
                valid_dset=valid_dset)

dset = get_dataset(args)
train_dset, valid_dset = dset['train_dset'], dset['valid_dset']

train_dl = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                      pin_memory=False)
valid_dl = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                      pin_memory=False)

if args.loss == "wsdr":
    loss = WeightedSDR()
else:
    raise NotImplementedError(f"unknown loss ({args.loss})")

if args.metric == "pesq":
    metric = PESQ()
else:
    def metric(output, bd):
        with torch.no_grad():
            return -loss(output, bd)

net = SourceSeparator(complex=args.complex, model_complexity=args.model_complexity, model_depth=args.model_depth, log_amp=args.log_amp, padding_mode=args.padding_mode).cuda()
print(net)

if args.multi_gpu:
    net = nn.DataParallel(net).cuda()

if args.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum)
else:
    raise ValueError(f"Unknown optimizer - {args.optimizer}")

if args.swa:
    steps_per_epoch = args.validation_interval
    optimizer = SWA(optimizer, swa_start=int(20) * steps_per_epoch, swa_freq=1 * steps_per_epoch)

if args.lr_decay >= 1 or args.lr_decay <= 0:
    scheduler = None
else:
    if args.optimizer == "swa":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, mode="max", patience=5, factor=args.lr_decay)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=args.lr_decay)

trainer = PinkBlack.trainer.Trainer(net,
                                    criterion=loss,
                                    metric=metric,
                                    train_dataloader=train_dl,
                                    val_dataloader=valid_dl,
                                    ckpt=args.ckpt,
                                    optimizer=optimizer,
                                    lr_scheduler=scheduler,
                                    is_data_dict=True,
                                    logdir="log_se")

trainer.train(step=args.num_step, validation_interval=args.validation_interval)

if args.swa:
    trainer.swa_apply(bn_update=True)
    trainer.train(1, phases=['val'])
