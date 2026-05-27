import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile as wf

fs = 44100
channels = 16

eps = 1e-12

nfft = 2048                 # FFT-storlek per tidsruta (ger frekvensupplösning)
noverlap = int(nfft * 0.75) # överlapp mellan rutor (ger mjukare tid)
window = np.hanning(nfft)   # "Gabor-ish" fönster

folder = 'thread_validation'
if os.path.exists(folder):
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.flac', '.mp3'))]
   
    for file in files:
        file_path = os.path.join(folder, file)
        print(file)
        try:
            samplerate, data = wf.read(file_path)

            duration = len(data)/fs

            plt.figure(figsize=(16, 8))

            Pxx, f, tt, im = plt.specgram(
                data,
                NFFT=nfft,
                Fs=fs,
                window=window,
                noverlap=noverlap,
                detrend='mean',
                scale='dB',     # visar i dB
                mode='psd'      # power spectral density
            )

            plt.title("Spektrogram (STFT / Gabor-lik tids-frekvens-analys)")
            plt.xlabel("Tid (s)")
            plt.ylabel("Frekvens (Hz)")
            plt.ylim(20, fs // 2)

            vmin = np.percentile(10 * np.log10(Pxx + eps), 5)
            vmax = np.percentile(10 * np.log10(Pxx + eps), 99)
            im.set_clim(vmin, vmax)

            plt.colorbar(im, label="Nivå (dB)")

            plt.show()
        except Exception as e:
            print(f"Error processing {file}: {e}")
   
else:
    print(f"Folder not found: {folder}")