# Deep Complex U-Net
---
Unofficial PyTorch Implementation of [Phase-Aware Speech Enhancement with Deep Complex U-Net](https://openreview.net/forum?id=SkeRTsAcYm), (H. Choi et al., 2018) 


## Architecture
---
(TO BE) 


## Train
---
1. Download Datasets:
- [https://datashare.is.ed.ac.uk/handle/10283/2791](https://datashare.is.ed.ac.uk/handle/10283/2791)

2. Separate each train / test wavs

3. Downsample wavs
```bash
# prerequisite : ffmpeg.
# sudo apt-get install ffmpeg (Ubuntu)
bash downsample.sh   # all wavs below $PWD will be converted to .flac, 16k samplerate
```

4. Train
```bash
python train_dcunet.py --batch_size 12 \
                       --train_signal /path/to/train/clean/speech/ \
                       --train_noise /path/to/train/noisy/speech/ \
                       --test_signal /path/to/test/clean/speech/ \
                       --test_noise /path/to/test/noisy/speech/ \
                       --num_step 50000 \
                       --validation_interval 500 \
                       --complex

# You can check other arguments from the source code. ( Sorry for the lack description. )                        
```

## Test
---
```bash
python estimate_directory.py --input_dir /path/to/noisy/speech/ \
                             --output_dir /path/to/estimate/dir/ \
                             --ckpt /path/to/saved/checkpoint.pth
```


## Results
---
| PESQ(cRMCn/cRMRn)   | Paper | Mine* |
| -------------------- | ----- | ---- |
| DCUNet - 10     |  **2.72**/2.51  | 3.34/**3.36**  |
| DCUNet - 20| **3.24**/2.74  | 3.35/**3.38** |

- *cRMCn* : Complex-valued input/Output
- *cRMRn* : Real-valued input/Output

Comparing the two(Paper's, Mine) values above is inappropriate for the following reasons:

- \* The PESQ I calculated is inaccurate because I calculated only one second from the beginning of the test set, not the whole wav. (It is not PESQ, It's pesudo-PESQ)

- \* The Architecture of model is slightly different from the original paper. (Such as kernel size of convolution filters) 


## Notes
---
- Log amplitute estimate has slightly worse performance than non-log amplitude
- Complex-valued network does not make the metric better..

## Sample Wavs
---
| Mixture | Estimated Speech | GT(Clean Speech) |
| --------|-----------|-------------|
|[mixture1.wav]()|[Estimated1.wav]()|[GroundTruth1.wav]()|
|[mixture2.wav]()|[Estimated2.wav]()|[GroundTruth2.wav]()|


## Contact
---
- Jongho Choi(sweetcocoa@snu.ac.kr / Seoul National Univ., ESTsoft )