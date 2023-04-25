import matplotlib.pyplot as plt
from scripts.mental_attention_state_detection_to_spikes import resample_signal , generate_spikes
import pandas as pd
import numpy as np
from matplotlib import mlab
from spicy import signal
import os
from tqdm import tqdm
from pathlib import Path

from snn.resonator import create_excitatory_inhibitory_resonator

path = "C:\\Program Files (x86)\project\\tdcsfog"  #local path for Shahar Halevy , change for your own

i = 0
fig, ax = plt.subplots()

clk_freq=15360
fs=128
channels = ['AccV', 'AccML', 'AccAP']
clk_resonators = {
    15360: ['0.6', '1.0', '1.39', '1.64', '1.93']

}
n_trials=803
n_channels = 3
n_resonators =5


###we create a resonator in each iteration maybe its wastefull, and we rather create a resonator and work on it till we finish with it.

with tqdm(total=n_channels * n_resonators * n_trials) as pbar:  #tqdm shows proccess precentage
    for trial in os.listdir(path):                              #each trial is each experiment in the dataset
        f = os.path.join(path, trial)
        data = pd.read_csv(f, index_col=0)
        npArray = np.array(data)
        for ch_i, ch_n in enumerate(channels):                  #3 channels in IMU
            ch_data = npArray[:, ch_i]                           #specific signal for the current channel
            data_resampled = resample_signal(clk_freq / 2, fs, ch_data)  #signal convert: frequency from sampled signal to clk frequency
            for clk_i, (clk_freq, list_of_f0) in enumerate(clk_resonators.items()): #we go through all clk_freq groups
                spikes_folder = f'../datasets/IMU_data/{trial}/{ch_n}/{clk_freq}'      #folder path for saving dataset
                if not os.path.exists(spikes_folder):
                    os.makedirs(spikes_folder)
                for f_i, f0 in enumerate(list_of_f0):                                #we go through all resonator frequency in each clk_freq group
                    pbar.set_description(f',trial: {trial}, ch: {ch_i}/3 clk {clk_i}/1 f:{f_i}/5')  #tqdm shows proccess precentage
                    pbar.update()
                    spikes_file = f'{spikes_folder}/{f0}.npz'
                    if Path(spikes_file).is_file():
                        continue
                    resonator = create_excitatory_inhibitory_resonator(                          #create the resonator to generate output of the signal through it
                        freq0=f0,
                        clk_freq=clk_freq)
                    resonator.log_out_spikes(-1)                                                 #the output is the last neuron's output
                    generate_spikes(resonator, data_resampled, spikes_file)            #save the output through current resonator in datasets



#
# path = "C:\\Program Files (x86)\project\\tdcsfog"
# first = True
# #N = 97077  # number of max elements
# #f="C:\\Users\ozzil\Desktop\project\kaggle_data\\00c4c9313d.parquet"
# i=0
# fig, ax = plt.subplots()
# mean = 0
# count = 0
# for filename in os.listdir(path):
#     count += 1
#     f = os.path.join(path,filename)
#     #print(f)
#     data = pd.read_csv(f,index_col=0)
#     #data = pd.read_parquet(f, engine='fastparquet')
#
#     #data = data['Torque'].astype(float).values
#     #print(data)
#     npArray=np.array(data)
#     sig1=npArray[:,0]
#     #print(sig1)
#     N = len(sig1)
#     mean += N
#     #sig2=npArray[:,1]
#     avg = sum(sig1)/N
#     sig1 = sig1 - avg
#     f, t, Sxx = signal.spectrogram(sig1, fs=128)
#     ax.pcolormesh(t, f, Sxx, shading='gouraud')
#
#     #sig3=npArray[:,2]
#     #while len(sig1) < 97077:
#         #np.append(sig1,0)
#         #np.append(sig2,0)
#         #np.append(sig3,0)
#     #if max < N:
#     #    max = N
#
#     #if first == True:
#         #first = False
#         #fft1_sum = np.fft.fft(sig1,N)
#         #fft2_sum = np.fft.fft(sig2, N)
#         #fft3_sum = np.fft.fft(sig3, N)
#         #fftfreq = np.fft.fftfreq(2359, 1 / 128)
#         #freq = np.linspace(0, 128, N)
#     #else:
#         #sig1_sum += sig1
#         #sig2_sum += sig2
#         #sig3_sum += sig3
#         #fft1_sum += np.fft.fft(sig1,2359)
#         #fft2_sum += np.fft.fft(sig2, N)
#         #fft3_sum += np.fft.fft(sig3, N)
#
#
# #print(len(fft3_sum))
# #figure, axis = plt.subplots(1, 3)
#
# #plt.ylabel("Frequency")
# #plt.xlabel("Time")
# #plt.plot(fftfreq[100:-100],abs(fft1_sum[100:-100]))
# #plt.plot(fftfreq,fft2)
# #plt.plot(fftfreq,abs(fft1_sum))
# #plt.plot(np.fft.fftfreq(N, 1 / 128),abs(np.fft.fft(sig1)))
#
# print(mean/count)
# #ax = plt.axes(projection='3d')
# #spec, freqs, t = mlab.specgram(sig1, Fs=128)
# #X, Y, Z = t[None, :], freqs[:, None], spec
# #ax.plot_surface(X, Y, Z, cmap='viridis')
# #ax.set_xlabel('time (s)')
# #ax.set_ylabel('frequencies (Hz)')
# #ax.set_zlabel('amplitude (dB)')
# #ax.set_ylim(0, 15)
#
# #axis[0]
#
# #plt.colorbar()
# #axis[1].
# #plt.specgram(np.fft.ifft(fft1_sum), Fs=128)
# plt.ylim(top=15)
# #plt.xlim(right=100)
# plt.yticks(np.arange(0,15,1))
# plt.hlines(np.arange(0,15,1),xmin=0,xmax=750,colors='w',linewidth=0.2)
#
# plt.show()