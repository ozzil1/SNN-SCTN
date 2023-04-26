import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab
from spicy import signal
import os

path = "../datasets/kaggle_data/tdcsfog/"

def plot_fft_summed_input(path):
    i = 0
    fig, ax = plt.subplots()
    mean = 0
    count = 0
    fft1_sum=[0]*4682
    for filename in os.listdir(path):
        f = os.path.join(path, filename)

        data = pd.read_csv(f, index_col=0, compression='gzip')
        # data.to_csv(path_or_buf=f'../datasets/kaggle_data/notype/{filename}', compression='gzip')
        # data = pd.read_parquet(f, engine='fastparquet')

        # data = data['Torque'].astype(float).values
        # print(data)
        npArray = np.array(data)
        sig1 = npArray[:, 0]
        N = len(sig1)
        sig2 = npArray[:, 1]
        avg = sum(sig1) / N
        sig1 = sig1 - avg
        # f, t, Sxx = signal.spectrogram(sig1, fs=128)
        # ax.pcolormesh(t, f, Sxx, shading='gouraud')
        sig3 = npArray[:, 2]

        #fft2_sum += np.fft.fft(sig2, N)
        #fft3_sum += np.fft.fft(sig3, N)
        if len(sig1) < 4682:
            np.concatenate((sig1,np.array([0]*(4682-len(sig1)))),axis=0)
        if len(sig1)>4682:
            sig1=sig1[0:4682]
        fft1_sum += np.fft.fft(sig1,4682)
        print(fft1_sum)
        # if max < N:
        #    max = N

        # if first == True:
        # first = False
        # fft1_sum = np.fft.fft(sig1,N)
        # fft2_sum = np.fft.fft(sig2, N)
        # fft3_sum = np.fft.fft(sig3, N)
    fftfreq = np.fft.fftfreq(4682, 1 / 128)
        # freq = np.linspace(0, 128, N)
        # else:
        # sig1_sum += sig1
        # sig2_sum += sig2
        # sig3_sum += sig3
        # fft1_sum += np.fft.fft(sig1,2359)
        # fft2_sum += np.fft.fft(sig2, N)
        # fft3_sum += np.fft.fft(sig3, N)

    # print(len(fft3_sum))
    # figure, axis = plt.subplots(1, 3)

    plt.ylabel("Amp")
    plt.xlabel("Frequency")
    plt.plot(fftfreq,abs(fft1_sum))
    plt.show()


def plot_spectogram_input(path):
    fig, ax = plt.subplots()
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        data = pd.read_csv(f, index_col=0, compression='gzip')
        npArray = np.array(data)
        sig1 = npArray[:, 0]
        f, t, Sxx = signal.spectrogram(sig1, fs=128)
        ax.pcolormesh(t, f, Sxx, shading='gouraud')

    plt.ylim(top=8)
    plt.yticks(np.arange(0, 25, 1))
    plt.xlim(left=0, right=100)
    plt.show()



# plt.plot(fftfreq,fft2)
# plt.plot(fftfreq,abs(fft1_sum))
# plt.plot(np.fft.fftfreq(N, 1 / 128),abs(np.fft.fft(sig1)))

# print(mean/count)
# ax = plt.axes(projection='3d')
# spec, freqs, t = mlab.specgram(sig1, Fs=128)
# X, Y, Z = t[None, :], freqs[:, None], spec
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('time (s)')
# ax.set_ylabel('frequencies (Hz)')
# ax.set_zlabel('amplitude (dB)')
# ax.set_ylim(0, 15)

# axis[0]

# plt.colorbar()
# #axis[1].
# #plt.specgram(np.fft.ifft(fft1_sum), Fs=128)
# #plt.ylim(top=15)
# plt.xlim(right=100)
# #plt.yticks(np.arange(0,15,1))
# #plt.hlines(np.arange(0,15,1),xmin=0,xmax=750,colors='w',linewidth=0.2)
# axis[2].specgram(fft3_sum, Fs=128)
# #plt.show()
