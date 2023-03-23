import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io.wavfile as wav


uploaded ="C:/Users/sarah/OneDrive/Documents/projet prt/Son_a.wav"


Son_AAA = wav.read(uploaded)

fe = Son_AAA[0]
Son_array = Son_AAA[1][10000:20000, 0]
ax_t = np.linspace(0, Son_array.size / fe, Son_array.size)

print(Son_array)
plt.plot(ax_t, Son_array)
plt.title("son d'origine")
plt.show()

filtered = sc.ndimage.gaussian_filter(Son_array, 90, mode="mirror")[200:-200]
fft_filtered = fft(filtered)
period = int(1 / (np.argmax(np.abs(fft_filtered[0:100])) / fft_filtered.size))
print(f"{period} échantillons par période (pour le traitement numérique)")
print(f"Période en secondes : {period/fe} (pour reconnaître la hauteur du son)")