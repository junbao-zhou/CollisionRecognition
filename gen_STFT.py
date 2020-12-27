from STFT import STFT
from glob import glob
import numpy as np
# audio_files = glob('./dataset/train/*/*/*.pkl')
audio_files = glob('./dataset/task2/test/*/*.pkl')

for file in audio_files:
    print(file)
    audio_map = STFT(file, False)
    np.save(file.replace('pkl', 'npy'), audio_map)
