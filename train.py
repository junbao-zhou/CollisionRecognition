import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import librosa

path = './dataset/train'

labels = ['061_foam_brick', 'salt_cylinder',  'stanley_screwdriver',  'toothpaste_box',  'whiteboard_spray',
          'green_basketball',  'shiny_toy_gun',  'strawberry',           'toy_elephant',    'yellow_block']
index = 1

for l in range(2, 3):
    files = glob.glob(f'{path}/{labels[l]}/1/*.pkl')
    # files = glob.glob(f'./dataset/task*/test/*/*.pkl')

    for f in files:
        print(f)
        data = np.load(f, allow_pickle=True)
        audio = data['audio']
        sample_rate = data['audio_samplerate']
        print(audio.shape)
        print(sample_rate)
        # plt.plot(data[:, 0])
        # plt.show()
        # if data.shape[0] != 176400:
        #     print(f'{f}   not match, shape : {data.shape}')
