import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from scipy.stats import trim_mean
from os import listdir
from os.path import isfile, join


def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N,mode='same')


def get_pca(path):
    data = pd.read_csv(path, header=None).values
    amp = data[:, 1:91]

    # Filter offset
    constant_offset = np.empty_like(amp)
    filtered_data = np.empty_like(amp)
    for i in range(1, len(amp[0])):
        constant_offset[:, i] = moving_average(amp[:, i], 4000)
    filtered_data = amp - constant_offset

    # Smoothing
    for i in range(1, len(amp[0])):
        filtered_data[:, i] = moving_average(filtered_data[:, i], 10)

    # PCA
    cov_mat2 = np.cov(filtered_data.T)
    eig_val2, eig_vec2 = np.linalg.eig(cov_mat2)
    idx = eig_val2.argsort()[::-1]
    eig_val2 = eig_val2[idx]
    eig_vec2 = eig_vec2[:, idx]
    pca_data2 = filtered_data.dot(eig_vec2)

    # Spectrogram of average PCA component
    plt.figure(figsize=(18, 10))
    new_data = np.zeros(shape=pca_data2[:, 0].shape)
    for i in range(6):
        if i != 4:
            continue
        new_data += pca_data2[:, i]
    #new_data /= 6.0
    new_data = pca_data2[:, 0]
    spec, f, t, im = plt.specgram(new_data, NFFT=256, Fs=1000, noverlap=1, cmap='jet', vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT) combined PCA components")
    plt.colorbar(im)
    # plt.xlim(0,20)
    plt.ylim(0, 80)
    #plt.show()

    return spec, f, t, data

if __name__ == '__main__':
    data_dir = 'Dataset/loc5'
    processed_dir = 'processed_data'
    csi_data_lst = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    spec, f, t, CSI = get_pca(csi_data_lst[0])

    tm = trim_mean(spec[3:8], 0, axis=0)    #0 to keep the entirety
    D = pd.Series(tm, t)
    tm_mva = runningMeanFast(D,11)
    plt.figure()
    plt.plot(tm_mva)

    plt.show()

    mvaPeaks, _ = find_peaks(tm_mva, distance=15, threshold=0)
    bins_mvaPeaks = t[mvaPeaks]
    bins_mvaPeaks = list(map(lambda x: int(x * 1000), bins_mvaPeaks))

    plt.plot(t[mvaPeaks], spec[mvaPeaks], 'x')
    plt.plot(t, spec)
    plt.show()


