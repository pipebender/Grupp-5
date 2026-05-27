import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wf

fs = 44100
channels = 16

samplerate, data = wf.read('./bot_validation/misc_20260522_114958.wav')

#print(data)

duration = len(data)/fs

eps = 1e-12

nfft = 2048                 # FFT-storlek per tidsruta (ger frekvensupplösning)
noverlap = int(nfft * 0.75) # överlapp mellan rutor (ger mjukare tid)
window = np.hanning(nfft)   # "Gabor-ish" fönster

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

#print(Pxx)
#print(f)
#print(tt)

plt.title("Spektrogram (STFT / Gabor-lik tids-frekvens-analys)")
plt.xlabel("Tid (s)")
plt.ylabel("Frekvens (Hz)")
plt.ylim(20, fs // 2)

# Gör färgskalan lite rimligare (valfritt men ofta bättre)
# Du kan kommentera bort om du vill.
vmin = np.percentile(10 * np.log10(Pxx + eps), 5)
vmax = np.percentile(10 * np.log10(Pxx + eps), 99)
im.set_clim(vmin, vmax)

plt.colorbar(im, label="Nivå (dB)")

plt.show()