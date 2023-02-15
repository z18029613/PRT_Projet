# -*- coding: utf-8 -*-
"""PRT_Vitesse.ipynb
Test codes pour PRT 19/01/2023
Lorenzo ROSADO
"""

#Libraries importées
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


 
plt.close(fig='all')

#Bloc d'importation du fichier 
uploaded = "Son_a.wav"

#Variable contenant le son : 
Son_AAA = wav.read(uploaded)
Son_freq = Son_AAA[0]           # fréquence du fichier
Son_array = Son_AAA[1]          # <!> il est de taille N x 2 car il y a deux canaux audios, oreille gauche et oreille droite.
Son_array_destruct=Son_array/2

#écrire le fichier wav et le télécharger avec méthode 1
file_test = wav.write("test_compression.wav", Son_freq*2, Son_array_destruct )
#Avec cette méthode le son sort presque détruit, on ne reconnait même pas le signal d'entrée



Son_AAA_fft = np.fft.fft(Son_array, norm="ortho")
N = int(len(Son_AAA_fft)/2)
x = np.arange(-N,N+1)

plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_array[:,0]))) , 'b',label="TF du son 'a'")
plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_array_destruct[:,0]))) ,'r',label="TF du son après compression")
plt.xlabel("frequence")
plt.ylabel("|S(f)|")


#Filtre passe bas
n = np.arange(0,len(Son_AAA_fft))
n_temp = np.zeros ( np.shape(Son_array) , dtype="int16")
n_temp[:,0],n_temp[:,1] = n , n 
n = n_temp

fe = Son_freq

Freq = np.linspace(-fe, fe, len(n))
F_temp = np.zeros ( np.shape(Son_array) , dtype="int16")
F_temp[:,0],F_temp[:,1] = Freq, Freq
Freq = F_temp

h = np.zeros ( np.shape(Son_array))# , dtype="int16")
freq_coupure = 400000
std = freq_coupure
coef = 1000

h[:,0] = 1/np.sqrt(2 * np.pi * std**2)  * np.exp(- Freq[:,0]**2 / (2* std**2) ) * coef
h[:,1] = 1/np.sqrt(2 * np.pi * std**2)  * np.exp(- Freq[:,1]**2 / (2* std**2) )  *coef



H = h #np.fft.fft(h)

Son_filtre = H * Son_AAA_fft 

plt.plot(x,np.abs(np.fft.fftshift(np.fft.fft(Son_filtre[:,0]))) , 'g',label="TF filtré par H")
plt.legend()
plt.figure(1)
plt.show()

plt.figure(2)
plt.plot(Freq ,  np.abs (    H ) , label = "|H|")
plt.legend()
plt.show()


plt.figure(3)

Son_AAA_postfiltre = np.array(  np.abs( np.fft.ifft (Son_filtre) ) , dtype = "int16")

plt.plot(Son_AAA_postfiltre,label="son post filtre")
plt.legend()
plt.show()

file_sonpostfiltre = wav.write("sonpostfiltre.wav", Son_freq , Son_AAA_postfiltre )






