# -*- coding: utf-8 -*-
"""PRT_Vitesse.ipynb
Test codes pour PRT 19/01/2023
Lorenzo ROSADO
"""

#Libraries importées
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.io.wavfile as wav
import io


#Bloc d'importation du fichier 
uploaded = "C:/Users/Domi/Desktop/AAA.wav"
#Variable contenant le son : 
Son_AAA = wav.read(uploaded)
Son_freq = Son_AAA[0]
Son_array = Son_AAA[1] # <!> il est de taille N x 2 car il y a deux canaux audios, oreille gauche et oreille droite.
Son_array_destruct=Son_array/2
#écrire le fichier wav et le télécharger avec méthode 1
test = wav.write("test.wav", Son_freq*2, Son_array_destruct )
#Avec cette méthode le son sort presque détruit, on ne reconnait même pas le signal d'entrée

Son_AAA_fft = np.fft.fft(Son_array, norm="ortho")
Son_AAA_selfconvol  = np.abs(np.fft.ifft(np.abs(Son_AAA_fft), norm="ortho"))

Son_AAA_selfconvol=Son_AAA_selfconvol/Son_AAA_selfconvol[0]


N = int(89019/2)
x = np.arange(-N,N+1)
plt.plot(x,np.log(np.abs(np.fft.fftshift(np.fft.fft(Son_array[:,0])))) , 'b')
plt.plot(x,np.log(np.abs(np.fft.fftshift(np.fft.fft(Son_array_destruct[:,0])))) ,'r')

plt.xlabel("frequence")
plt.ylabel("log (|S(f)|")


plt.show()
