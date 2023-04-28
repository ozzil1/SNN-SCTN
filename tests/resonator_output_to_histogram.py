import os
import numpy as np
import pandas as pd
from spicy import signal
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt

IMU_data = f'../datasets/IMU_data'

channel = 'AccAP'


def load_preprocessed_spikes(trial, channel):
    def append_zeros_to_fix_size(filename, size=6):
        # remove .npz
        filename = filename[:-4]
        filename = ('0' * (size - len(filename))) + filename
        return filename

    resonators_dir = f'{IMU_data}/{trial}/{channel}'
    return {
        append_zeros_to_fix_size(f0):
            np.load(f'{resonators_dir}/{clk_freq}/{f0}')['spikes'].astype(np.int8)
        for clk_freq in os.listdir(resonators_dir)
        for f0 in os.listdir(f'{resonators_dir}/{clk_freq}')
    }


def spikes_to_bins(spikes, window=195_000):
    num_bins = (len(spikes) // window) + 1
    bins = np.zeros(num_bins, dtype=np.int32)
    for j in range(num_bins):
        bins[j] = np.sum(spikes[j * window:(j + 1) * window])
    return bins


def all_spikes2bins(full_spikes_data, window=195_000):
    return {
        f0: spikes_to_bins(spikes, window)
        for f0, spikes in full_spikes_data.items()
    }


def normalize_arr(arr):
    # return arr
    return (arr - min(arr)) / (max(arr) - min(arr))


def spikes_data2eeg_bands(spikes_data):
    return {
        # '0. Delta': sum(normalize_arr(spikes_data[f])
        #                      for f in ['00.657', '01.523', '02.120', '02.504', '03.490']) / 5,
        '1. Theta': sum(normalize_arr(spikes_data[f])
                        for f in ['04.604', '05.755', '06.791', '08.000']) / 4,
        '2. Alpha': sum(normalize_arr(spikes_data[f])
                        for f in ['08.058', '09.065', '10.072', '11.885', '14.000']) / 5,
        '3. Beta': sum(normalize_arr(spikes_data[f])
                       for f in ['15.108', '17.266', '19.424', '21.583', '25.468']) / 5,
        '4. Gamma': sum(normalize_arr(spikes_data[f])
                        # for f in ['36.259', '40.791', '45.324', '53.482', '63.000']) / 5
                        for f in ['36.259', '40.791']) / 2
    }


def spikes_data2imu_bands(spikes_data):
    return {
        'AccV': sum(normalize_arr(spikes_data[f])
                    for f in ['0000.6', '0001.0', '001.39', '001.64', '001.93']) / 5

    }


def normalize_columns(arr):
    """
    Normalizes each column of a 2D NumPy array so that the minimum value in each column is 0 and the maximum is 1.

    Args:
    - arr: A 2D NumPy array with shape (n, k).

    Returns:
    - A normalized version of arr with the same shape.
    """
    # Find the minimum and maximum values for each column
    col_mins = np.min(arr, axis=0)
    col_maxs = np.max(arr, axis=0)

    # Make sure there are no divisions by zero
    col_ranges = np.where(col_maxs == col_mins, 1, col_maxs - col_mins)

    # Normalize each column by subtracting its minimum and dividing by its range
    return (arr - col_mins) / col_ranges


def spikes_dict2spectogram(
        output_spikes,
):
    spikes_heatmap = pd.DataFrame.from_dict(output_spikes)
    spikes_heatmap = spikes_heatmap.reindex(sorted(spikes_heatmap.columns), axis=1).T
    spikes_heatmap_data = spikes_heatmap.to_numpy()

    spikes_heatmap_data = spikes_heatmap_data
    return normalize_columns(spikes_heatmap_data)


def plot_heatmap(fig,ax,heatmap_data, y_labels, annotate, title=None):
    #fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        heatmap_data,
        cmap='jet',
        aspect='auto',
        origin='lower',
        vmin=np.min(heatmap_data),
        vmax=np.max(heatmap_data),
    )
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    # Loop over data dimensions and create text annotations.
    if annotate:
        for i in range(len(heatmap_data)):
            for j in range(len(heatmap_data[i])):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.4f}',
                               rotation=90,
                               ha="center", va="center", color="white")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    #fig.colorbar(im, ax=ax, label='Interactive colorbar')
    #plt.show()


##################################   plot by plot_heatmap  ##################################
# fig, ax = plt.subplots()
#
# for trial in os.listdir(IMU_data):
#     example_spikes_channel = load_preprocessed_spikes(trial, channel)
#
#     IMU_bands = spikes_data2imu_bands(all_spikes2bins(example_spikes_channel, window=500))
#     IMU_bands = IMU_bands[channel]
#     if (len(IMU_bands)>1500):
#         IMU_bands=IMU_bands[0:1500]
#     IMU_bands = np.trim_zeros(IMU_bands,'b')
#     print(IMU_bands)
#     plot_heatmap(fig,ax,[IMU_bands], IMU_bands.keys(), annotate=False, title='Spikes spectogram')
#
# # fig.colorbar(im, ax=ax, label='Interactive colorbar')
# # plt.show()
#
# plt.ylim(top=8)
# #plt.xlim(top=150)
# plt.yticks(np.arange(0,15,1))
# #plt.hlines(np.arange(0,15,1),xmin=0,xmax=150,colors='w',linewidth=0.2)
# plt.show()

#####################################################################################################

########################   plot spectogram of data not averaging the frequencies  ###################
def plot_spectogram(IMU_data,channel):
    fig, ax = plt.subplots()

    for trial in os.listdir(IMU_data):
        example_spikes_channel = load_preprocessed_spikes(trial, channel)
        for f in ['0000.6', '0001.0', '001.39', '001.64', '001.93']:
            IMU_bands = (all_spikes2bins(example_spikes_channel, window=60))[f]
            #IMU_bands = np.trim_zeros(IMU_bands,'b')
            print(np.array(IMU_bands))

            f, t, Sxx = signal.spectrogram(IMU_bands, fs=(15360/2)/60)
            ax.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim(top=8)
    plt.yticks(np.arange(0,25,1))
    plt.xlim(left=0,right=100)
    plt.show()

#####################################################################################################


########################  plot fft summed ################################
def plot_fft_summed(IMU_data,channel):
    fft1_sum=[0]*4682
    for trial in os.listdir(IMU_data):
        example_spikes_channel = load_preprocessed_spikes(trial, channel)
        for f in ['0000.6', '0001.0', '001.39', '001.64', '001.93']:
            IMU_bands = (all_spikes2bins(example_spikes_channel, window=122))[f]
            if (len(IMU_bands)>4682):
                IMU_bands=IMU_bands[0:4682]
            if len(IMU_bands) < 4682:
                np.concatenate((IMU_bands, np.array([0] * (4682 - len(IMU_bands)))), axis=0)
            fft1_sum += np.fft.fft(IMU_bands, 4682)
            print(fft1_sum)

    fftfreq = np.fft.fftfreq(4682, 1/ 128)
    plt.ylabel("Amp")
    plt.xlabel("Frequency")
    plt.plot(fftfreq,abs(fft1_sum))
    plt.show()

def plot_single_signal_spectogram(IMU_data,trial,channel):
    fig, ax = plt.subplots()
    example_spikes_channel = load_preprocessed_spikes(trial, channel)
    for f in ['0000.6', '0001.0', '001.39', '001.64', '001.93']:
        IMU_bands = (all_spikes2bins(example_spikes_channel, window=60))[f]

        f, t, Sxx = signal.spectrogram(IMU_bands, fs=(15360/2)/60)
        ax.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim(top=8)
    plt.yticks(np.arange(0,25,1))
    #plt.xlim(left=0,right=100)
    plt.show()

trial='0a89f859b5.csv'
plot_single_signal_spectogram(IMU_data,trial,channel)
