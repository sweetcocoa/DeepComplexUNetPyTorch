from pypesq import pesq
import os, glob
import soundfile
from tqdm import tqdm

refs = sorted(glob.glob("path/to/clean/*.flac"))
evals = sorted(glob.glob("path/to/estimated/*.flac"))

results = dict()
for i, (ref, eval) in tqdm(enumerate(zip(refs, evals))):
    assert os.path.basename(ref) == os.path.basename(eval)
    y, sr = soundfile.read(ref)
    y_hat, sr = soundfile.read(eval)

    results[os.path.basename(ref)] = pesq(y, y_hat, sr)

# print(results)
print(sum(results.values())/len(results))
