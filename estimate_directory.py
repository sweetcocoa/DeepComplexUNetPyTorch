"""
Usage ::
python estimate_directory.py --input_dir /path/to/input/wavs/ --output_dir /path/to/output/wavs/ --ckpt ckpt/ckpt.pth

input_dir (폴더 안의 폴더 recursively 포함)의 *.wav, *.flac을 노이즈 제거하여
output_dir 에 출력한다.

알려진 문제점
1) 많은 노이즈가 제대로 제거되지 않는 문제
2) 느린 문제
"""

import os, glob
import soundfile
import numpy as np
import torch
import json
from tqdm import tqdm
from DCUNet.constant import *
from DCUNet.source_separator import SourceSeparator
from DCUNet.inference import any_audio_inference
from easydict import EasyDict

import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/path/to/data/wav_dir/")
    parser.add_argument('--output_dir', default="/path/to/data/wav_dir/")
    parser.add_argument('--ckpt', default="ckpt/190628_False_mix_20_sz60.pth")
    args = parser.parse_args()

    args = EasyDict(args.__dict__)

    ckpt = args.ckpt
    args.sequence_length = 16384 # 고정

    model_spec = sorted(glob.glob(ckpt + "*.args"))[-1]
    with open(model_spec) as f:
        specs = EasyDict(json.load(f))

    args.update(specs)
    args.ckpt = ckpt  # 저장된 args의 ckpt로 덮어써지므로.

    if not hasattr(args, "padding_mode"):
        print("No 'padding_mode' is specified, 'zeros' will be used as padding_mode")
        args.padding_mode = "zeros"

    return args

args = get_arg()
input_files = []
input_files.extend(glob.glob(args.input_dir + "/**/*.wav", recursive=True))
input_files.extend(glob.glob(args.input_dir + "/**/*.flac", recursive=True))

net = SourceSeparator(complex=args.complex,
                      log_amp=args.log_amp,
                      model_complexity=args.model_complexity,
                      model_depth=args.model_depth,
                      padding_mode=args.padding_mode
                      ).cuda()

net.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
net.eval()

os.makedirs(args.output_dir, exist_ok=True)

for file in tqdm(sorted(input_files)):
    y_hat = any_audio_inference(file, net, sequence_length=args.sequence_length, normalize=True).transpose()
    output_file = file.replace(args.input_dir, args.output_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    soundfile.write(output_file, y_hat, SAMPLE_RATE)