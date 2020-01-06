'''
Preprocesses the consolidated data, outputs numpy tensors
'''
import os
import sys
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


data_dir = 'Dataset/loc5'
processed_dir = 'processed_data'
csi_data_lst = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def visualize(path1, load_data=True, data=None):
    # data import
    if load_data:
        data = pd.read_csv(path1, header=None).values
    amp = data[:, 1:91]

    # plt
    fig = plt.figure(figsize=(18, 10))
    ax1 = plt.subplot(311)
    plt.imshow(amp[:, 0:29].T, interpolation="nearest", aspect="auto", cmap="jet")
    ax1.set_title("Antenna1 Amplitude")
    plt.colorbar()

    ax2 = plt.subplot(312)
    plt.imshow(amp[:, 30:59].T, interpolation="nearest", aspect="auto", cmap="jet")
    ax2.set_title("Antenna2 Amplitude")
    plt.colorbar()

    ax3 = plt.subplot(313)
    plt.imshow(amp[:, 60:89].T, interpolation="nearest", aspect="auto", cmap="jet")
    ax3.set_title("Antenna3 Amplitude")
    plt.colorbar()
    #plt.show()

    # Initializing valiables
    constant_offset = np.empty_like(amp)
    filtered_data = np.empty_like(amp)

    # Calculating the constant offset (moving average 4 seconds)
    for i in range(1, len(amp[0])):
        constant_offset[:, i] = moving_average(amp[:, i], 4000)

    # Calculating the filtered data (substract the constant offset)
    filtered_data = amp - constant_offset

    # Smoothing (moving average 0.01 seconds)
    for i in range(1, len(amp[0])):
        filtered_data[:, i] = moving_average(filtered_data[:, i], 10)
    # Calculate correlation matrix (90 * 90 dim)

    cov_mat2 = np.cov(filtered_data.T)
    # Calculate eig_val & eig_vec
    eig_val2, eig_vec2 = np.linalg.eig(cov_mat2)
    # Sort the eig_val & eig_vec
    idx = eig_val2.argsort()[::-1]
    eig_val2 = eig_val2[idx]
    eig_vec2 = eig_vec2[:,idx]
    # Calculate H * eig_vec
    pca_data2 = filtered_data.dot(eig_vec2)

    xmin = 0
    xmax = 200000

    # PCA Plots
    fig3 = plt.figure(figsize = (18,20))
    ax_pca = []
    k = 611
    for i in range(6):
        ax = plt.subplot(k + i)
        plt.plot(pca_data2[xmin:xmax,i])
        #plt.plot(pca_data2[2500:17500,0])
        ax.set_title("PCA {} component".format(i+1))
        ax_pca.append(ax)

    #plt.show()

    k = 611
    plt.figure(figsize=(18, 30))
    Pxx_arr, freqs_arr, bins_arr, im_arr = [], [], [], []

    # Spectrogram(STFT)
    for i in range(6):
        plt.subplot(k + i)
        Pxx, freqs, bins, im = plt.specgram(pca_data2[:, i], NFFT=256,
                                            Fs=1000, noverlap=1, cmap="jet", vmin=-100, vmax=20)
        plt.xlabel("Time[s]")
        plt.ylabel("Frequency [Hz]")
        plt.title("Spectrogram(STFT) Component {}".format(i + 1))
        plt.colorbar(im)
        # plt.xlim(0,20)
        plt.ylim(0, 80)

        Pxx_arr.append(Pxx)
        freqs_arr.append(freqs)
        bins_arr.append(bins)
        im_arr.append(im)

    #plt.show()

    #average component
    plt.figure(figsize=(18,10))
    new_data = np.zeros(shape=pca_data2[:,0].shape)
    for i in range(6):
        new_data += pca_data2[:, i]
    new_data /= 6.0
    a,b,c, im = plt.specgram(new_data, NFFT=256, Fs=1000, noverlap=1, cmap='jet', vmin=-100, vmax=20)
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram(STFT) combined PCA components")
    plt.colorbar(im)
    # plt.xlim(0,20)
    plt.ylim(0, 80)

    plt.figure(figsize=(18, 10))
    ax = plt.subplot(111)
    #    ax.magnitude_spectrum(pca_data2[:,0], Fs=1000, scale='dB', color='C1')
    ax.magnitude_spectrum(pca_data2[5000:7500, 0], Fs=1000, color='C1')
    plt.xlim(0, 50)
    plt.ylim(0, 10)
    plt.show()

    return a, b, c, pca_data2


#print(csi_data_lst) #circ, lr, x, ud is the order
Pxx5, freqs5, bins5, _ = visualize(path1=join(data_dir, csi_data_lst[0]))

#Trim_mean 0.1, for main freqs components from ~10 Hz to 27 Hz
#Very important paramenter

tm = trim_mean(Pxx5[3:8], 0, axis=0) #0 to keep the entirety

plt.plot(bins5, tm)
D = pd.Series(tm, bins5)

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N,mode='same')

#calculate moving average/moving mean not rolling
tm_mva = runningMeanFast(D,11)
plt.figure()
plt.plot(tm_mva)

plt.show()

### MVA PEAKS ###
mvaPeaks, _ = find_peaks(tm_mva, distance=15, threshold=0)
bins_mvaPeaks=bins5[mvaPeaks]
bins_mvaPeaks = list(map(lambda x: int(x * 1000), bins_mvaPeaks))
CSI = pd.read_csv(join(data_dir, csi_data_lst[0]), header=None).values


def get_CSI_partition(CSI, peaks, split):
    partition = np.empty(shape=(len(peaks), split*2, CSI.shape[1]))
    for i in range(len(peaks)): # we assume that data collection started early enough and ended soon enough,
        # so we won't make an exception for first and last peaks in case the window goes beyond the first and
        # last indices. Otherwise, just disregard first and last peak, losing data
        partition[i] = CSI[(peaks[i]-split):(peaks[i]+split)]
    return partition


avg_split = 4000
CSI_part = get_CSI_partition(CSI, bins_mvaPeaks[1:], int(avg_split/2))
print(CSI_part.shape)