import subprocess
import os
from easydict import EasyDict
import GPUtil
import sys, time

gpu = GPUtil.getAvailable(limit=4, excludeID=[6,7])

print("gpu available : ", gpu)

args = EasyDict(dict(gpu="2",
                     batch_size=12,
                     train_signal="/data/jongho/data/2019challenge/dcunet/clean_speech/train/",
                     train_noise="/data/jongho/data/2019challenge/dcunet/noisy_noise/train/",
                     test_signal="/data/jongho/data/2019challenge/ss/reference/16k/clean_testset_wav/",
                     test_noise="/data/jongho/data/2019challenge/ss/reference/16k/noisy_testset_wav/",
                     sequence_length=16384,
                     num_step=40000,
                     validation_interval=500,
                     num_workers=12,
                     ckpt="unet/ckpt.pth",
                     model_complexity=45,
                     lr=0.01,
                     num_signal=0,
                     num_noise=0,
                     optimizer="adam",
                     lr_decay=0.5,
                     momentum=0,
                     multi_gpu=False,
                     complex=True,
                     model_depth=10,
                     swa=False,
                     loss="wsdr",
                     log_amp=False,
                     metric="pesq",
                     train_dataset="se",
                     valid_dataset="se",
                     preload=False,
                     padding_mode="reflect"))

se_y_train = "/data/jongho/data/2019challenge/ss/reference/16k/clean_trainset_28spk_wav/"
se_x_train = "/data/jongho/data/2019challenge/ss/reference/16k/noisy_trainset_28spk_wav/"

# mix_y_train = "/data/jongho/data/2019challenge/ss/clean_speech/train/"
# mix_x_train = "/data/jongho/data/2019challenge/ss/noisy_noise/train/"

mix_y_train = "/data/jongho/data/2019challenge/dcunet/clean_speech/train/"
mix_x_train = "/data/jongho/data/2019challenge/dcunet/demand/train/"

# mix_y_train = "/data/jongho/data/2019challenge/ss/dataset/train/speech/"
# mix_x_train = "/data/jongho/data/2019challenge/ss/dataset/train/noise/"

for model_complexity in [45, 90]:
    for model_depth in [10, 20]:
        # skip cnfig
        if model_complexity == 90 and model_depth == 10:
            continue
        if model_complexity == 45 and model_depth == 20:
            continue

        for complex in [False, True]:
            for log in [False]:
                for train_dataset in ['se']:
                    for optimizer, lr in [('adam', 0.01)]:
                        for padding_mode in ['zeros']:
                            while not gpu:
                                sleep_sec = 600
                                print(f"no gpu available, sleep {sleep_sec}s...")
                                time.sleep(sleep_sec)
                                gpu = GPUtil.getAvailable(limit=4, excludeID=[6,7])

                            command = [f"/miniconda/bin/python",  f"{os.getcwd()}/train_dcunet.py"]
                            args.train_dataset = train_dataset

                            if train_dataset == "se":
                                args.train_signal = se_y_train
                                args.train_noise = se_x_train
                            else:
                                args.train_signal = mix_y_train
                                args.train_noise = mix_x_train

                            args.padding_mode = padding_mode
                            args.model_depth = model_depth
                            args.gpu = str(gpu.pop())
                            args.model_complexity = model_complexity
                            args.ckpt = f"demand_experiment_report/190717_{log}_dp{model_depth}_{train_dataset}_sz{model_complexity}_{padding_mode}_comp_{complex}.pth"
                            args.optimizer = optimizer
                            args.lr = lr

                            for k,v in args.items():
                                if isinstance(v, bool):
                                    pass
                                else:
                                    command.append(f"--{k}")
                                    command.append(f"{v}")

                            if log:
                                command.append("--log_amp")

                            if args.preload:
                                command.append("--preload")

                            if complex:
                                command.append("--complex")

                            print("command : {", command, "}")
                            subprocess.Popen(command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            time.sleep(1)
                        # exit()

time.sleep(86400*10)