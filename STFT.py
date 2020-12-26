import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt


def STFT(audio_file, is_plot, freq_length, time_length):
    data = np.load(audio_file, allow_pickle=True)
    audio = data['audio']
    sample_rate = data['audio_samplerate']

    stft_result = []
    for i in range(4):
        audio_resample = ss.resample(audio[:, i], audio.shape[0] // 4)
        stft_re = ss.stft(audio_resample, nperseg=512, noverlap=384)[2]
        stft_result.append(np.abs(stft_re))
    stft_result = np.array(stft_result)
    stft_result /= np.max(stft_result)
    # print(np.unravel_index(np.argmax(stft_result), stft_result.shape))

    time_mid = int(stft_result.shape[2] / 2)
    time_left = time_mid - 100
    time_right = time_left + time_length
    audio_map = stft_result[:, 0:freq_length, time_left:time_right]

    if is_plot:
        print(audio_file)
        print(f'audio shape = {audio.shape}')
        plt.plot(audio[:, 3])
        plt.show()
        print(f'audio_resample.shape = {audio_resample.shape}')
        plt.plot(audio_resample)
        plt.show()
        print(f'stft_result.shape = {stft_result.shape}')
        print(time_left)
        print(time_right)
        print(f'audio_map.shape = {audio_map.shape}')

        plt.imshow(audio_map[0], cmap='gray')
        plt.show()
        plt.imshow(audio_map[1], cmap='gray')
        plt.show()
        plt.imshow(audio_map[2], cmap='gray')
        plt.show()
        plt.imshow(audio_map[3], cmap='gray')
        plt.show()

    return audio_map
