import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab
from spicy import signal
import os


path = "C:\\Users\ozzil\Desktop\project\kaggle_data\\tdcsfog"
first = True
#N = 97077  # number of max elements
#f="C:\\Users\ozzil\Desktop\project\kaggle_data\\00c4c9313d.parquet"
i=0
fig, ax = plt.subplots()
mean = 0
count = 0
for filename in os.listdir(path):
    count += 1
    f = os.path.join(path,filename)
    #print(f)
    data = pd.read_csv(f,index_col=0)
    #data = pd.read_parquet(f, engine='fastparquet')

    #data = data['Torque'].astype(float).values
    #print(data)
    npArray=np.array(data)
    sig1=npArray[:,0]
    N = len(sig1)
    mean += N
    #sig2=npArray[:,1]
    avg = sum(sig1)/N
    sig1 = sig1 - avg
    f, t, Sxx = signal.spectrogram(sig1, fs=128)
    ax.pcolormesh(t, f, Sxx, shading='gouraud')
    #sig3=npArray[:,2]
    #while len(sig1) < 97077:
        #np.append(sig1,0)
        #np.append(sig2,0)
        #np.append(sig3,0)
    #if max < N:
    #    max = N

    #if first == True:
        #first = False
        #fft1_sum = np.fft.fft(sig1,N)
        #fft2_sum = np.fft.fft(sig2, N)
        #fft3_sum = np.fft.fft(sig3, N)
        #fftfreq = np.fft.fftfreq(2359, 1 / 128)
        #freq = np.linspace(0, 128, N)
    #else:
        #sig1_sum += sig1
        #sig2_sum += sig2
        #sig3_sum += sig3
        #fft1_sum += np.fft.fft(sig1,2359)
        #fft2_sum += np.fft.fft(sig2, N)
        #fft3_sum += np.fft.fft(sig3, N)


#print(len(fft3_sum))
#figure, axis = plt.subplots(1, 3)

#plt.ylabel("Frequency")
#plt.xlabel("Time")
#plt.plot(fftfreq[100:-100],abs(fft1_sum[100:-100]))
#plt.plot(fftfreq,fft2)
#plt.plot(fftfreq,abs(fft1_sum))
#plt.plot(np.fft.fftfreq(N, 1 / 128),abs(np.fft.fft(sig1)))

print(mean/count)
#ax = plt.axes(projection='3d')
#spec, freqs, t = mlab.specgram(sig1, Fs=128)
#X, Y, Z = t[None, :], freqs[:, None], spec
#ax.plot_surface(X, Y, Z, cmap='viridis')
#ax.set_xlabel('time (s)')
#ax.set_ylabel('frequencies (Hz)')
#ax.set_zlabel('amplitude (dB)')
#ax.set_ylim(0, 15)

#axis[0]

#plt.colorbar()
#axis[1].
#plt.specgram(np.fft.ifft(fft1_sum), Fs=128)
plt.ylim(top=15)
#plt.xlim(right=100)
plt.yticks(np.arange(0,15,1))
plt.hlines(np.arange(0,15,1),xmin=0,xmax=750,colors='w',linewidth=0.2)
#axis[2].specgram(fft3_sum, Fs=128)
plt.show()