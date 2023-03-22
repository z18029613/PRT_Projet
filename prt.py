# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import pywt
import numpy as np
from numpy.fft import fft,ifft, fftshift
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io.wavfile as wav


uploaded ="C:/Users/sarah/OneDrive/Documents/projet prt/Son_a.wav"

Son_AAA = wav.read(uploaded)

Son_freq = Son_AAA[0]
Son_array = Son_AAA[1][:,0] # <!> il serait de taille N x 2 car il y a deux canaux audios, oreille gauche et oreille droite.

print(Son_array)
plt.plot(Son_array)
plt.title("son d'origine")
plt.show()

def check_periodic(data):
    periodic = False
    for i in range(2, len(data)//2):
        for j in range(len(data) - i):
            if not np.allclose(data[j], data[j+i], rtol=200):
                periodic = False
                break
            periodic = True
        if periodic:
            return i
    return -1

# Vérification de la périodicité et récupération de la valeur

period = check_periodic(Son_array[40100:41000])
if period == -1:
    print(f'Le signal n\'est pas périodique')
else:
    print(f'Le signal a une période de {period}')

plt.plot(Son_array[40100:41000])
plt.title("son d'origine")
plt.show()

# def ss_ech_ond (signal, mere, seuil):
#     coeffs = pywt.wavedec(signal ,mere )
#     coeffs[1:] = [ pywt.threshold(i, seuil, mode = 'soft', substitute= 0) for i in coeffs[1:]]
#     faster_signal= pywt.waverec(coeffs, mere)
#     return faster_signal


# signal2 = ss_ech_ond(Son_array, 'coif1', 0.5)
# print(signal2)
# plt.plot(signal2, color='r')
# plt.title("son accéléré")
# plt.show()
    
    

