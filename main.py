import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io.wavfile as wav
import scipy.ndimage as nd

def save_audio(file_path, sampling_rate, audio_signal):
    wav.write(file_path, sampling_rate, audio_signal.astype(np.int16))

uploaded = "./Son_o.wav"

Son_AAA = wav.read(uploaded)

fe = Son_AAA[0]
Son_array = Son_AAA[1][:, 0]
ax_t = np.linspace(0, Son_array.size / fe, Son_array.size)

print(Son_array)
plt.plot(ax_t, Son_array)
plt.title("son d'origine")
plt.show()

filtered = nd.gaussian_filter(Son_array, 90, mode="mirror")[200:-200]
fft_filtered = fft(filtered)
period = int(1 / ((10+np.argmax(np.abs(fft_filtered[10:1000]))) / fft_filtered.size))
print(f"{period} échantillons par période (pour le traitement numérique)")
print(f"Période en secondes : {period/fe} (pour reconnaître la hauteur du son)")

# Conservation d'une période sur deux
Son_conserve = np.zeros(Son_array.size // 2)
for i in range(0, Son_conserve.size - period, period):
    print(i, period)
    Son_conserve[i:i+period] = Son_array[2*i:2*i+period]

# Filtre passe-bas pour régulariser
order = 4  # Ordre du filtre
cutoff_freq = 0.2  # Fréquence de coupure du filtre passe-bas
b, a = sc.signal.butter(order, cutoff_freq)
Son_filtre = sc.signal.lfilter(b, a, Son_conserve)

# Enregistrement du nouveau son
output_path = "./Son_accelere.wav"
save_audio(output_path, fe, Son_filtre)

# Affichage des signaux
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(ax_t, Son_array)
plt.title("Son d'origine")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

#ax_t[:Son_conserve.size],
plt.subplot(3, 1, 2)
plt.plot( Son_conserve[10000:11000])
plt.title("Signal conservé avec une période sur deux")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot( Son_filtre[10000:11000])
plt.title("Signal régularisé avec filtre passe-bas")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")


sig=np.zeros(1000)
sig[499]+=1
c,d=sc.signal.butter(order,cutoff_freq)
fft_filtre=fftshift(fft(sc.signal.lfilter(c,d,sig)))
plt.figure()
plt.plot(np.abs( fft_filtre))
plt.axis("off")
plt.title("fonction de transfert filtre passe-bas")


plt.tight_layout()
plt.show()