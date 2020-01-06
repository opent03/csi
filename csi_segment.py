import os
import numpy as np
import scipy as sp
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
# this code is used to segment csi csv file. hope it works
# miao.wang.2@emt.inrs.ca

FS = 1000

COMPONENT_INDEX_FFT = 5 # fifith PCA component to find the PEAKS in FFT
ENTIRE_LEN = 2 * FS     # 2 second data

LABEL = None
VERIFICATION_FOLDER = "~/Dekstop/csi-verify/"
SEGMENTED_DATA_FOLDER = "~/Desktop/csi-segmented/"


def gen_label(possib):
    while True:
        for v in possib:
            yield v

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

[i for i in map(check_path, [VERIFICATION_FOLDER, SEGMENTED_DATA_FOLDER])]

def low_pass_filter(fs=FS, cutoff=30, order=7): # parameter impacts on the performace
    b, a = butter(order, 0.5 * cutoff * fs, btype='low', analog=False)
    return (b,a)

LPF =  lambda arr: lfilter(*low_pass_filter(), arr)

def read_csi_csv(csv_file_path):
    # keep amplitude only
    df = pd.read_csv(csv_file_path, sep=',', header=None)
    # time index 0, amp 1-90, phase 91-180
    return df[:, 1:91].to_numpy() # change here could have the phase info 


def apply_pca(amp_arr, pca_componet4peak=COMPONENT_INDEX_FFT):
    # shape (num_t,90)
    win_size = 5
    moving_avg_window = np.ones(win_size) / win_size
    tmp_arr = np.zeros_like(amp_arr)
    # filter out high freq noise
    for c in range(amp_arr.shape[1]):
        tmp_arr[:, c] = np.convolve(amp_arr[:,c], moving_avg_window, 'same')
    # center
    center = np.mean(tmp_arr, axis=0)
    tmp_arr = tmp_arr - center
    # decomposition
    covariance = np.cov(tmp_arr.T)
    eig_val, eig_vec = np.linalg.eig(covariance)
    # sort by importance
    idx = np.argsort(eig_val)[::-1]
    # eig_val_sorted = eig_val[idx]
    eig_vec_sorted = eig_vec[:, idx]
    data = np.dot(tmp_arr, eig_vec_sorted)
    pxx, freqs, t, im = plt.specgram(data[:,pca_componet4peak], NFFT=256, Fs=Fs, noverlap=1, cmap="jet", vmin=-100,vmax=20)
    return pxx, freqs, t

def cmp_peaks(pxx, freqs, t, label=LABEL):
    # plot and save peak images for later verification
    smooth = np.ones([0.1, 0.2, 0.3, 0.2, 0.1])
    new_pxx = np.convolve(pxx, smooth, 'same')
    peaks, _ = find_peaks(new_pxx, distance=15, threshold=0.01)  # parameters impact on the performance
    plt.plot(t, new_pxx)
    time_index = t[peaks]
    plt.plot(time_index, new_pxx[peaks], 'r+')
    figname = label + '.png'
    plt.savefig(os.path.join(VERIFICATION_FOLDER, figname))
    if len(time_index) < 23:
        print("warning: the number of peaks seems wrong. checking {}".format(figname))
    if len(time_index) % 2 !=0:
        print('Warning: the number of peaks seems wrong. checking {}.\n ignore this if the label is circle or cross')
    return time_index 


def iterate_folder_path(father_path, room_folders, loc_folders, activity_folders):
    tmp = "{}+{}+{}".format
    path_tmp = "{}/{}/{}/{}/".format
    for r in room_folders:
        for l in loc_folders:
            for act in activity_folders:
                yield (tmp(r, l, act), path_tmp(father_path, r, l, act))
   
def merge_CSV_to_numpy(des_folder, csv_file_list):
    numpy_container = [] # it consumes a lot memeory

    for csv_file in csv_file_list:
        tmp = read_csi_csv(os.path.join(des_folder, csv_file))
        numpy_container.append(tmp) # only amp
    return np.concatenate(numpy_container, axis=0) # really big 


def segment_single_folder(label, des_folder_path):
    LABEL = label
    # merge all the local csv files
    allcsv = [i for i in os.listdir(des_folder_path) if i.endswith('.csv')]
    allcsv.sort() # sort by the time stamp
    entire_numpy = merge_CSV_to_numpy(des_folder_path, allcsv)
    # apply pca 
    pxx, freqs, t = apply_pca(entire_numpy, pca_componet4peak=COMPONENT_INDEX_FFT)
    # find peak index
    peak_row_index = cmp_peaks(pxx, freqs, t,  label=LABEL)
    
    # apply low pass filter 
    if entire_numpy.shape[1] != 90:
        print('ERROR, the DIM of feature should be 90')
    
    for c in range(entire_numpy.shape[1]):
        feature = entire_numpy[:, c]
        entire_numpy[:, c] = LPF(feature)

    # got time index
    # segment data case by case 

    y = label.split('+')[-1]
    if y == 'circle' or y == 'cross':
        pk_num = 0
        for s,e in zip(peak_row_index[:-1], peak_row_index[1:]):
            length = ENTIRE_LEN // 2
            if s - length  < 0 or s + length >= e:
                print("discard sample center #peak={}, index ={}, label={}".format(pk_num, s, LABEL))
            else:
                tmp = entire_numpy[s-length:s+length, :]
                np.savez_compressed('{}.npz'.format(label), tmp)
            pk_num += 1
    elif y == 'up-down':
        label_gen = gen_label(['up', 'down'])
        pk_num = 0
        for s,e in zip(peak_row_index[:-1], peak_row_index[1:]):
            length = ENTIRE_LEN // 2
            real_label = next(label_gen)
            real_label = "{}+{}".format("+".join(label.split('+')[:-1]), real_label) 
            if s - length  < 0 or s + length >= e:
                print("discard sample center #peak={}, index ={}, label={}".format(pk_num, s, real_label))
            else:
                tmp = entire_numpy[s-length:s+length, :]
                np.savez_compressed('{}.npz'.format(real_label), tmp)
            pk_num += 1
    elif y == 'left-right':
        label_gen = gen_label(['left', 'right'])
        pk_num = 0
        for s,e in zip(peak_row_index[:-1], peak_row_index[1:]):
            length = ENTIRE_LEN // 2
            real_label = next(label_gen)
            real_label = "{}+{}".format("+".join(label.split('+')[:-1]), real_label) 
            if s - length  < 0 or s + length >= e:
                print("discard sample center #peak={}, index ={}, label={}".format(pk_num, s, real_label))
            else:
                tmp = entire_numpy[s-length:s+length, :]
                np.savez_compressed('{}.npz'.format(real_label), tmp)
            pk_num += 1
    else:
        print('FATAL ERROR')


def segment(father_path):
    room_folders = ('lab', 'meeting')
    loc_folders = ['loc-{}'.format(i) for i in range(1,17)]
    activity_folders = ('circle', 'cross', 'up-down', 'left-right')

    label_path_iter = iterate_folder_path(father_path, room_folders, loc_folders, activity_folders)
    for label, des_folder_path in label_path_iter:
        segment_single_folder(label, des_folder_path)


if __name__ == "__main__":
    """
        father_folder:
                    |------- lab
                    |         |
                    |         |----- loc1
                    |         |
                    |         |----  loc2
                    |                 |
                    |                 |--------- circle 
                    |                 |
                    |                 |--------- cross
                    |                 |
                    |                 |--------- left-right
                    |                 |
                    |                 |--------- up-down
                    | ..................................
                    |------- meeting  (similar arrangement goes here)
    """
    father_folder = "."
    segment(father_folder) # monitor  warnings and tune parameter based on the generted peak figure 
    # the re-segmentation could run in case by case if the warnings don't present frequently 
    # when to re-do for 1 folder, use 
    # segment_single_folder(label, des_folder_path)
    